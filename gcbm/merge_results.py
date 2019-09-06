# Python 2.7 compatibility.
from future.standard_library import install_aliases
install_aliases()

import logging
import sys
import os
from fnmatch import fnmatch
from argparse import ArgumentParser
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy import bindparam
from sqlalchemy import String
from sqlalchemy import MetaData
from sqlalchemy import Table
from sqlalchemy import select
from sqlalchemy import insert
from sqlalchemy import BLANK_SCHEMA

def find_results_databases(root_path, file_pattern=None, recursive=False):
    results_databases = []
    if recursive:
        for dir, subdirs, files in os.walk(root_path):
            for file in filter(lambda f: fnmatch(f, file_pattern), files):
                results_databases.append(os.path.abspath(os.path.join(dir, file)))
    else:
        for file in os.listdir(root_path):
            full_file_path = os.path.abspath(os.path.join(root_path, file))
            if os.path.isfile(full_file_path) and fnmatch(file, file_pattern):
                results_databases.append(full_file_path)
    
    logging.info("Found {} results database{} to merge.".format(
        len(results_databases),
        "s" if len(results_databases) != 1 else ""))
        
    return results_databases

def find_project_classifiers(conn):
    with conn.begin():
        results = conn.execute(text("SELECT * FROM _v_age_indicators LIMIT 1"))
        classifiers = [
            col for col in results.keys()
            if col.lower() not in ("year, unfccc_land_class, age_range, area")
        ]
            
    return classifiers
        
def merge_results_tables(from_conn, to_conn):
    logging.info("Merging reporting tables into output database...")
    md = MetaData()
    md.reflect(bind=from_conn,
               only=lambda table_name, _: table_name.startswith("v_")
                                          and "density" not in table_name)
    output_md = MetaData(bind=to_conn, schema=None)
    with to_conn.begin():
        for fqn, table in md.tables.items():
            logging.info("  {}".format(fqn))
            output_table_name = "_{}".format(table.name)
            table.tometadata(output_md, schema=None, name=output_table_name)
            output_table = Table(output_table_name, output_md, keep_existing=True)
            output_table.create(checkfirst=True)
            
            batch = []
            for i, row in enumerate(from_conn.execute(select([table]))):
                batch.append({k: v for k, v in row.items()})
                if i % 10000 == 0:
                    to_conn.execute(insert(output_table), batch)
                    batch = []
            
            if batch:
                to_conn.execute(insert(output_table), batch)

def condense_results_tables(conn):
    classifiers = ",".join(find_project_classifiers(conn))

    sql = []
    for table, aggregate_cols, value_col in [
        ("v_flux_indicator_aggregates", ["unfccc_land_class, indicator, year, age_range"],                                     "flux_tc"),
        ("v_flux_indicators",           ["unfccc_land_class, indicator, year, disturbance_code, disturbance_type, age_range"], "flux_tc"),
        ("v_pool_indicators",           ["unfccc_land_class, indicator, year, age_range"],                                     "pool_tc"),
        ("v_stock_change_indicators",   ["unfccc_land_class, indicator, year, age_range"],                                     "flux_tc"),
    ]:
        sql.append(
            """
            CREATE TABLE {table} AS
            SELECT
                {classifiers},
                {aggregate_cols},
                SUM(t.area) AS area,
                SUM(t.{value_col}) AS {value_col},
                SUM(t.{value_col}) / SUM(area) AS {value_col}_per_ha
            FROM _{table} t
            GROUP BY
                {classifiers},
                {aggregate_cols}
            """.format(table=table, classifiers=classifiers, value_col=value_col,
                       aggregate_cols=",".join(aggregate_cols)))
    
    sql.append(
        """
        CREATE TABLE v_age_indicators AS
        SELECT
            {classifiers},
            year,
            unfccc_land_class,
            age_range,
            SUM(t.area) AS area
        FROM _v_age_indicators t
        GROUP BY
            {classifiers},
            year,
            unfccc_land_class,
            age_range
        """.format(classifiers=classifiers))

    sql.append(
        """
        CREATE TABLE v_disturbance_indicators AS
        SELECT
            {classifiers},
            unfccc_land_class,
            year,
            disturbance_code,
            disturbance_type,
            pre_dist_age_range,
            post_dist_age_range,
            SUM(t.dist_area) AS dist_area,
            SUM(t.dist_product) AS dist_product,
            SUM(t.dist_product) / SUM(t.dist_area) AS dist_product_per_ha
        FROM _v_disturbance_indicators t
        GROUP BY
            {classifiers},
            unfccc_land_class,
            year,
            disturbance_code,
            disturbance_type,
            pre_dist_age_range,
            post_dist_age_range
        """.format(classifiers=classifiers))

    sql.append(
        """
        CREATE TABLE v_total_disturbed_areas AS
        SELECT
            {classifiers},
            unfccc_land_class,
            year,
            disturbance_code,
            disturbance_type,
            SUM(t.dist_area) AS dist_area
        FROM _v_total_disturbed_areas t
        GROUP BY
            {classifiers},
            unfccc_land_class,
            year,
            disturbance_code,
            disturbance_type
        """.format(classifiers=classifiers))
        
    for table in ("v_flux_indicators", "v_flux_indicator_aggregates", "v_stock_change_indicators"):
        sql.append(
            """
            CREATE TABLE {table}_density AS
            SELECT
               total_flux.indicator,
               CAST(total_area.year AS INTEGER) AS year,
               CAST(COALESCE(total_flux.flux_tc, 0) AS REAL) AS flux_tc,
               CAST(COALESCE(total_flux.flux_tc / total_area.area, 0) AS REAL) AS flux_tc_per_ha
            FROM (
               SELECT
                   year,
                   SUM(ai.area) AS area
                FROM v_age_indicators ai
                GROUP BY year
            ) AS total_area
            LEFT JOIN (
                SELECT
                    indicator,
                    year,
                    SUM(flux.flux_tc) AS flux_tc
                FROM {table} flux
                GROUP BY
                    indicator,
                    year
            ) AS total_flux
                ON total_area.year = total_flux.year
            ORDER BY
                total_flux.indicator,
                total_flux.year
            """.format(table=table))

    logging.info("Condensing merged results tables...")
    with conn.begin():
        for stmt in sql:
            conn.execute(text(stmt))

        md = MetaData()
        md.reflect(bind=conn, only=lambda table_name, _: table_name.startswith("_"))
        for _, table in md.tables.items():
            table.drop(bind=conn)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(message)s",
                        datefmt="%m/%d %H:%M:%S")

    parser = ArgumentParser(description="Produce reporting tables from raw GCBM results. For connection strings, "
                                        "see https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls")

    parser.add_argument("merge_root",   help="root directory where output databases to be merged are stored")
    parser.add_argument("file_pattern", help="file pattern for results databases to merge")
    parser.add_argument("output_db",    help="connection string for the database to copy final reporting tables to")
    parser.add_argument("--recursive",  help="scan subdirectories for results databases", action="store_true")
    parser.set_defaults(recursive=False)
    args = parser.parse_args()

    if os.path.exists(args.output_db):
        os.remove(args.output_db)

    output_db_engine = create_engine("sqlite:///{}".format(args.output_db))
    output_conn = output_db_engine.connect()
    
    for results_db in find_results_databases(args.merge_root, args.file_pattern, args.recursive):
        results_db_engine = create_engine("sqlite:///{}".format(results_db))
        results_conn = results_db_engine.connect()
        merge_results_tables(results_conn, output_conn)
    
    condense_results_tables(output_conn)
