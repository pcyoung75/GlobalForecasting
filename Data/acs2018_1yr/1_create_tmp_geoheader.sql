DROP TABLE IF EXISTS acs2018_1yr.tmp_geoheader;
CREATE TABLE acs2018_1yr.tmp_geoheader (
	all_fields varchar
)
WITH (autovacuum_enabled = FALSE, toast.autovacuum_enabled = FALSE)
;
