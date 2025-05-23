import logging
import os
import shutil
import string
import pyodbc
import pandas as pd
import numpy as np
import sqlite3
from logging import FileHandler
from logging import StreamHandler
from argparse import ArgumentParser

class Flux:

    def __init__(self, source_pool, destination_pool, proportion):
        self.source_pool = source_pool
        self.destination_pool = destination_pool
        self.proportion = float(proportion)

    def __str__(self):
        return f"{self.source_pool} to {self.destination_pool}: {self.proportion}"


class DisturbanceMatrix:

    def __init__(self, name):
        self.name = name
        self.disturbance_type = name
        self.fluxes = []
        self.ecoboundaries = []
        
    def __str__(self):
        return f"{self.name}\n  " + "\n  ".join((str(flux) for flux in self.fluxes))
        
    def add_flux(self, flux):
        self.fluxes.append(flux)
        
    def associate_ecoboundary(self, ecoboundary):
        self.ecoboundaries.append(ecoboundary)

        
def df_to_xls(row, col):
    return f"{string.ascii_uppercase[col]}{row + 1}"


def scan_for_matrices(files):
    matrices = []
    associations = {}

    matrix_header = ["s", None, None]
    matrix_flux   = ["s",  "s",  "n"]

    associations_table_header    = [ "s", None, None]
    associations_item_header     = [ "s",  "s",  "s"]
    associations_matrix_item_row = [None,  "s",  "s"]
    associations_eco_item_row    = [None, None,  "s"]

    for file in files:
        sheets = pd.read_excel(file, header=None, sheet_name=None)
        for sheet, df in sheets.items():
            # Find all the populated cells and whether they're a string or numeric type
            # to simplify detecting the disturbance matrix definitions.
            search_mask = df.map(
                lambda val: (
                    "s" if isinstance(val, str)
                    else "n" if isinstance(val, float) or isinstance(val, int)
                    else None
                ) if pd.notnull(val) else None)

            checked = pd.DataFrame(False, columns=df.columns, index=df.index)

            for r, row in search_mask.iterrows():
                for c, val in enumerate(row):
                    # Skip previously checked cells.
                    if checked.loc[r, c]:
                        continue

                    checked.loc[r, c] = True

                    # Skip cells too close to the end of the sheet to form a matrix definition.
                    if c > len(row) - 3 or r == len(search_mask):
                        continue
                    
                    '''
                    Matrices are defined as a name in the top-left cell followed by at least
                    one carbon flux, i.e.:
                    <name>      <blank>    <blank>
                    <source>    <sink>     <proportion>
                    '''
                    is_matrix = np.all(search_mask.loc[r:r, c:c + 2].values == matrix_header) \
                        and search_mask.loc[r + 1:r + 1, c:c + 2].size > 0 \
                        and np.all(search_mask.loc[r + 1:r + 1, c:c + 2].values == matrix_flux)
                        
                    '''
                    Disturbance matrix associations are defined as a single cell with the value
                    "Associations" followed by rows of the format:
                    <disturbance type>    <disturbance matrix>    <ecoboundary>
                                                                  <ecoboundary>
                    
                    If a matrix doesn't appear in the associations table, a disturbance type of
                    the same name gets created and associated with all ecoboundaries.
                    '''
                    is_associations_table = np.all(search_mask.loc[r:r, c:c + 2].values == associations_table_header) \
                        and np.all(search_mask.loc[r + 1:r + 1, c: c + 2].values == associations_item_header)
                        
                    if is_matrix:
                        # Found a matrix - now collect all the data.
                        matrix_name = df.loc[r, c]
                        logging.info(f"Matrix found in {sheet} at {df_to_xls(r, c)}: {matrix_name}")
                        checked.loc[r:r, c:c + 2] = True
                        
                        matrix = DisturbanceMatrix(matrix_name)
                        for matrix_row_idx in range(r + 1, len(search_mask)):
                            if not np.all(search_mask.loc[matrix_row_idx:matrix_row_idx, c:c + 2].values == matrix_flux):
                                # The matrix is complete when the from/to/proportion pattern ends.
                                break

                            matrix.add_flux(Flux(*df.loc[matrix_row_idx:matrix_row_idx, c: c + 2].values.tolist()[0]))
                            checked.loc[matrix_row_idx:matrix_row_idx, c:c + 2] = True
                        
                        matrices.append(matrix)
                    elif is_associations_table:
                        logging.info(f"DM associations found in {sheet} at {df_to_xls(r, c)}")
                        checked.loc[r:r, c:c + 2] = True
                        
                        current_dist_type = None
                        current_matrix = None
                        for row_idx in range(r + 1, len(search_mask)):
                            if not (
                                np.all(search_mask.loc[row_idx:row_idx, c:c + 2].values == associations_item_header)
                                or np.all(search_mask.loc[row_idx:row_idx, c:c + 2].values == associations_matrix_item_row)
                                or np.all(search_mask.loc[row_idx:row_idx, c:c + 2].values == associations_eco_item_row)
                            ):
                                # The association table is complete when the content patterns end.
                                break

                            dist_type, matrix, ecoboundary = df.loc[row_idx:row_idx, c: c + 2].values.tolist()[0]
                            current_dist_type = dist_type if pd.notnull(dist_type) else current_dist_type
                            current_matrix = matrix if pd.notnull(matrix) else current_matrix
                            
                            if current_matrix not in associations:
                                associations[current_matrix] = {}
                            
                            associations[current_matrix]["disturbance_type"] = current_dist_type
                            
                            if "ecoboundaries" not in associations[current_matrix]:
                                associations[current_matrix]["ecoboundaries"] = []
                            
                            associations[current_matrix]["ecoboundaries"].append(ecoboundary)
                            checked.loc[row_idx:row_idx, c:c + 2] = True
    
    for matrix in matrices:
        dm_associations = associations.get(matrix.name)
        if not dm_associations:
            continue
        
        matrix.disturbance_type = dm_associations["disturbance_type"]
        for ecoboundary in dm_associations["ecoboundaries"]:
            matrix.associate_ecoboundary(ecoboundary)
        
    return matrices

def load_aidb_matrices(matrices, aidb_path):
    connect_string = "DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={}"
    conn = pyodbc.connect(connect_string.format(aidb_path))
    try:
        for matrix in matrices:
            insert_aidb_matrix(conn, matrix)
    finally:
        if conn:
            conn.commit()
            conn.close()

def insert_aidb_matrix(conn, matrix):
    cur = conn.cursor()
    if cur.execute("SELECT TOP 1 * FROM tbldm WHERE name = ?", [matrix.name]).fetchone() is not None:
        logging.info(f"Matrix '{matrix.name}' already exists - skipping.")
        return

    logging.info(f"Inserting matrix: \n{matrix}")

    cur.execute(
        """
        INSERT INTO tbldm (dmid, name, description, dmstructureid)
        SELECT TOP 1 MAX(dmid) + 1, ?, ?, 2
        FROM tbldm
        """, [matrix.name, matrix.name])
    
    if cur.execute(
        "SELECT TOP 1 * FROM tbldisturbancetypedefault WHERE disttypename = ?",
        [matrix.disturbance_type]
    ).fetchone() is None:
        cur.execute(
            """
            INSERT INTO tbldisturbancetypedefault (disttypeid, disttypename, description)
            SELECT TOP 1 MAX(disttypeid) + 1, ?, ?
            FROM tbldisturbancetypedefault
            """, [matrix.disturbance_type, matrix.disturbance_type])
    
    dm_association_sql = \
        """
        INSERT INTO tbldmassociationdefault (
            defaultdisturbancetypeid, defaultecoboundaryid, annualorder, dmid,
            name, description)
        SELECT
            disttypeid, ecoboundaryid, 1, dmid, name & '-' & ecoboundaryname,
            name & '-' & ecoboundaryname
        FROM tbldisturbancetypedefault dt,
             tblecoboundarydefault eco,
             tbldm dm
        WHERE dt.disttypename = ?
            AND dm.name = ?
        """
        
    params = [matrix.disturbance_type, matrix.name]
    
    if matrix.ecoboundaries:
        logging.info(f"Inserting {matrix.name} for {', '.join(matrix.ecoboundaries)}")
        dm_association_sql += f" AND ecoboundaryname IN ({','.join('?' * len(matrix.ecoboundaries))})"
        params.extend(matrix.ecoboundaries)
    else:
        logging.info(f"Inserting {matrix.name} with universal associations")
        
    cur.execute(dm_association_sql, params)
    
    for flux in matrix.fluxes:
        if cur.execute(
            "SELECT TOP 1 * FROM tblsourcename WHERE dmstructureid = 2 AND description = ?",
            [flux.source_pool]
        ).fetchone() is None:
            logging.info(f"WARNING: Adding new pool '{flux.source_pool}' to tblSourceName")
            cur.execute(
                """
                INSERT INTO tblsourcename (dmstructureid, row, description)
                SELECT TOP 1 2, MAX(row) + 1, ?
                FROM tblsourcename
                WHERE dmstructureid = 2
                """, [flux.source_pool])
                
        if cur.execute(
            "SELECT TOP 1 * FROM tblsinkname WHERE dmstructureid = 2 AND description = ?",
            [flux.destination_pool]
        ).fetchone() is None:
            logging.info(f"WARNING: Adding new pool '{flux.destination_pool}' to tblSinkName")
            cur.execute(
                """
                INSERT INTO tblsinkname (dmstructureid, column, description)
                SELECT TOP 1 2, MAX(column) + 1, ?
                FROM tblsinkname
                WHERE dmstructureid = 2
                """, [flux.destination_pool])

        cur.execute(
            """
            INSERT INTO tbldmvalueslookup (dmid, dmrow, dmcolumn, proportion)
            SELECT dmid, row, column, ?
            FROM tbldm, tblsourcename src, tblsinkname snk
            WHERE name = ?
                AND src.dmstructureid = 2
                AND src.description = ?
                AND snk.dmstructureid = 2
                AND snk.description = ?
            """, [flux.proportion, matrix.name, flux.source_pool, flux.destination_pool])

def load_cbm_defaults_matrices(matrices, aidb_path):
    with sqlite3.connect(aidb_path) as conn:
        for matrix in matrices:
            insert_cbm_defaults_matrix(conn, matrix)

        conn.commit()
    
def insert_cbm_defaults_matrix(conn, matrix):
    if conn.execute(
        "SELECT * FROM disturbance_matrix_tr WHERE locale_id = 1 AND name = ? LIMIT 1",
        [matrix.name]
    ).fetchone() is not None:
        logging.info(f"Matrix '{matrix.name}' already exists - skipping.")
        return

    logging.info(f"Inserting matrix: \n{matrix}")

    conn.execute(
        """
        INSERT INTO disturbance_matrix (id)
        SELECT MAX(id) + 1
        FROM disturbance_matrix
        """)
    
    conn.execute(
        """
        INSERT INTO disturbance_matrix_tr (id, disturbance_matrix_id, locale_id, name, description)
        SELECT MAX(dm_tr.id) + 1, MAX(dm.id), 1, ?, ?
        FROM disturbance_matrix dm,
             disturbance_matrix_tr dm_tr
        """, [matrix.name, matrix.name])
    
    if conn.execute(
        "SELECT * FROM disturbance_type_tr WHERE locale_id = 1 AND name = ? LIMIT 1",
        [matrix.disturbance_type]
    ).fetchone() is None:
        conn.execute(
            """
            INSERT INTO disturbance_type (id)
            SELECT MAX(id) + 1
            FROM disturbance_type
            """)

        conn.execute(
            """
            INSERT INTO disturbance_type_tr (id, disturbance_type_id, locale_id, name, description)
            SELECT MAX(dt_tr.id + 1), MAX(dt.id), 1, ?, ?
            FROM disturbance_type dt,
                 disturbance_type_tr dt_tr
            """, [matrix.disturbance_type, matrix.disturbance_type])
    
    dm_association_sql = \
        """
        INSERT INTO disturbance_matrix_association (
            spatial_unit_id, disturbance_type_id, disturbance_matrix_id)
        SELECT
            spu.id, dt.disturbance_type_id, dm.disturbance_matrix_id
        FROM disturbance_type_tr dt,
             disturbance_matrix_tr dm,
             spatial_unit spu
             INNER JOIN eco_boundary_tr eco
                ON spu.eco_boundary_id = eco.eco_boundary_id
        WHERE dt.locale_id = 1 AND dt.name = ?
            AND dm.locale_id = 1 AND dm.name = ?
            AND eco.locale_id = 1
        """
        
    params = [matrix.disturbance_type, matrix.name]
    
    if matrix.ecoboundaries:
        logging.info(f"Inserting {matrix.name} for {', '.join(matrix.ecoboundaries)}")
        dm_association_sql += f" AND eco.name IN ({','.join('?' * len(matrix.ecoboundaries))})"
        params.extend(matrix.ecoboundaries)
    else:
        logging.info(f"Inserting {matrix.name} with universal associations")
        
    conn.execute(dm_association_sql, params)
    
    for flux in matrix.fluxes:
        for pool in (flux.source_pool, flux.destination_pool):
            if conn.execute(
                "SELECT * FROM pool_tr WHERE locale_id = 1 AND name = ?",
                [flux.source_pool]
            ).fetchone() is None:
                raise RuntimeError(f"Pool '{pool}' does not exist in AIDB")

        conn.execute(
            """
            INSERT INTO disturbance_matrix_value (
                disturbance_matrix_id, source_pool_id, sink_pool_id, proportion)
            SELECT disturbance_matrix_id, p_src.id, p_snk.id, ?
            FROM disturbance_matrix_tr dm,
                 pool_tr p_src,
                 pool_tr p_snk
            WHERE dm.name = ?
                AND p_src.locale_id = 1
                AND p_src.name = ?
                AND p_snk.locale_id = 1
                AND p_snk.name = ?
            """, [flux.proportion, matrix.name, flux.source_pool, flux.destination_pool])


if __name__ == "__main__":
    parser = ArgumentParser(description="Add DMs to AIDB")
    parser.add_argument("aidb_path", type=os.path.abspath, help="Path to original AIDB")
    parser.add_argument("output_path", type=os.path.abspath, help="Path to copy of AIDB to create")
    parser.add_argument("matrix_paths", type=os.path.abspath, nargs="+", help="Path to disturbance matrix spreadsheet(s)")
    args = parser.parse_args()
    
    if os.path.exists(args.output_path):
        os.remove(args.output_path)
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    shutil.copyfile(args.aidb_path, args.output_path)

    log_path = os.path.join(os.path.dirname(args.output_path), "add_dms.log")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        FileHandler(log_path, mode="w"),
        StreamHandler()
    ])

    matrices = scan_for_matrices(args.matrix_paths)
    if args.output_path.endswith(".mdb"):
        load_aidb_matrices(matrices, args.output_path)
    else:
        load_cbm_defaults_matrices(matrices, args.output_path)
