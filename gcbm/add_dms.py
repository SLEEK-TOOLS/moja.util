import os
import shutil
import string
import pyodbc
import pandas as pd
import numpy as np
from argparse import ArgumentParser

class Flux:
    
    def __init__(self, source_pool, destination_pool, proportion):
        self.source_pool = source_pool
        self.destination_pool = destination_pool
        self.proportion = proportion


class DisturbanceMatrix:

    def __init__(self, name):
        self.name = name
        self.fluxes = []
        
    def add_flux(self, flux):
        self.fluxes.append(flux)

        
def df_to_xls(row, col):
    return f"{string.ascii_uppercase[col]}{row + 1}"


def scan_for_matrices(file):
    matrix_header = ["s", None, None]
    matrix_flux   = ["s",  "s",  "n"]

    sheets = pd.read_excel(file, header=None, sheet_name=None)
    for sheet, df in sheets.items():
        # Find all the populated cells and whether they're a string or numeric type
        # to simplify detecting the disturbance matrix definitions.
        search_mask = df.applymap(
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
                if c > len(row) - 3 or r == len(search_mask) - 1:
                    continue
                
                # Matrices are defined as a name in the top-left cell followed by at least
                # one carbon flux, i.e.:
                # <name>      <blank>    <blank>
                # <source>    <sink>     <proportion>
                is_matrix = np.all(search_mask.loc[r:r, c:c + 2].values == matrix_header) \
                    and np.all(search_mask.loc[r + 1:r + 1, c: c + 2].values == matrix_flux)
                    
                if not is_matrix:
                    continue

                # Found a matrix - now collect all the data.
                matrix_name = df.loc[r, c]
                print(f"Matrix found in {sheet} at {df_to_xls(r, c)}: {matrix_name}")
                checked.loc[r:r, c:c + 2] = True
                
                matrix = DisturbanceMatrix(matrix_name)
                for matrix_row_idx in range(r + 1, len(search_mask)):
                    if not np.all(search_mask.loc[matrix_row_idx:matrix_row_idx, c:c + 2].values == matrix_flux):
                        # The matrix is complete when the from/to/proportion pattern ends.
                        break

                    matrix.add_flux(Flux(*df.loc[matrix_row_idx:matrix_row_idx, c: c + 2].values.tolist()[0]))
                    checked.loc[matrix_row_idx:matrix_row_idx, c:c + 2] = True
                
                yield matrix


def insert_matrix(conn, matrix):
    cur = conn.cursor()
    for sql in (
        """
        INSERT INTO tbldm (dmid, name, description, dmstructureid)
        SELECT TOP 1 MAX(dmid) + 1, ?, ?, 2
        FROM tbldm
        """,
        """
        INSERT INTO tbldisturbancetypedefault (disttypeid, disttypename, description)
        SELECT TOP 1 MAX(disttypeid) + 1, ?, ?
        FROM tbldisturbancetypedefault
        """,
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
    ):
        cur.execute(sql, [matrix.name, matrix.name])
    
    for flux in matrix.fluxes:
        if cur.execute(
            "SELECT TOP 1 * FROM tblsourcename WHERE dmstructureid = 2 AND description = ?",
            [flux.source_pool]
        ).fetchone() is None:
            cur.execute(
                """
                INSERT INTO tblsourcename (dmstructureid, column, description)
                SELECT TOP 1 2, MAX(column) + 1, ?
                FROM tblsourcename
                WHERE dmstructureid = 2
                """, [flux.source_pool])
                
        if cur.execute(
            "SELECT TOP 1 * FROM tblsinkname WHERE dmstructureid = 2 AND description = ?",
            [flux.destination_pool]
        ).fetchone() is None:
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


if __name__ == "__main__":
    parser = ArgumentParser(description="Add DMs to AIDB")
    parser.add_argument("matrix_path", type=os.path.abspath, help="Path to disturbance matrix spreadsheet")
    parser.add_argument("aidb_path", type=os.path.abspath, help="Path to original AIDB")
    parser.add_argument("output_path", type=os.path.abspath, help="Path to copy of AIDB to create")
    args = parser.parse_args()
    
    if os.path.exists(args.output_path):
        os.remove(args.output_path)
        
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    shutil.copyfile(args.aidb_path, args.output_path)

    connect_string = "DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={}"
    conn = pyodbc.connect(connect_string.format(args.output_path))
    try:
        for matrix in scan_for_matrices("residue_dms_test.xlsx"):
            insert_matrix(conn, matrix)
    finally:
        if conn:
            conn.commit()
            conn.close()
