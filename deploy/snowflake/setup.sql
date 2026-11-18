-- One-time setup for `fourthdown snowflake build` (phase 11). Run as ACCOUNTADMIN in a
-- Snowsight worksheet; a 30-day trial account is enough.
--
-- 1. Make a key pair on your machine (keep rsa_key.p8 out of the repo):
--      openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_key.p8 -nocrypt
--      openssl rsa -in rsa_key.p8 -pubout -out rsa_key.pub
-- 2. Paste rsa_key.pub's body (between the BEGIN/END lines, no line breaks) below, run this.
-- 3. In .env: SNOWFLAKE_ACCOUNT (Snowsight > account menu > "Connect a tool"),
--    SNOWFLAKE_USER=FOURTHDOWN, SNOWFLAKE_PRIVATE_KEY_PATH=/path/to/rsa_key.p8.
-- 4. fourthdown snowflake build && fourthdown snowflake parity

USE ROLE ACCOUNTADMIN;

-- The smallest warehouse, suspended after a minute idle: a full build is a few minutes of
-- XSMALL, a weekly incremental one well under one.
CREATE WAREHOUSE IF NOT EXISTS FOURTHDOWN_WH
    WAREHOUSE_SIZE = XSMALL
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE;

-- A hard ceiling on what this project can spend.
CREATE RESOURCE MONITOR IF NOT EXISTS FOURTHDOWN_MONITOR
    WITH CREDIT_QUOTA = 10
    TRIGGERS ON 80 PERCENT DO NOTIFY
             ON 100 PERCENT DO SUSPEND;
ALTER WAREHOUSE FOURTHDOWN_WH SET RESOURCE_MONITOR = FOURTHDOWN_MONITOR;

-- The build creates its databases (the live one and the clone it builds in), so it owns
-- them, which is what SWAP WITH and DROP need. It is granted nothing else.
CREATE ROLE IF NOT EXISTS FOURTHDOWN;
GRANT USAGE ON WAREHOUSE FOURTHDOWN_WH TO ROLE FOURTHDOWN;
GRANT CREATE DATABASE ON ACCOUNT TO ROLE FOURTHDOWN;
-- So the databases are visible from your own login in Snowsight.
GRANT ROLE FOURTHDOWN TO ROLE SYSADMIN;

-- A service user: key-pair only, no password to leak.
CREATE USER IF NOT EXISTS FOURTHDOWN
    TYPE = SERVICE
    DEFAULT_ROLE = FOURTHDOWN
    DEFAULT_WAREHOUSE = FOURTHDOWN_WH
    RSA_PUBLIC_KEY = 'PASTE_RSA_KEY_PUB_BODY_HERE';
GRANT ROLE FOURTHDOWN TO USER FOURTHDOWN;
