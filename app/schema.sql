-- schema.sql
-- Run this once against your KRM database to create the members table.
-- On Windows, from a command prompt with MySQL's bin folder on PATH:
--   mysql -u krm_app -p krm_db < schema.sql

CREATE TABLE IF NOT EXISTS members (
    id                                  INT AUTO_INCREMENT PRIMARY KEY,
    full_name                           VARCHAR(255) NOT NULL,
    age                                 INT NOT NULL,
    province                            VARCHAR(64) NOT NULL,
    residence                           VARCHAR(16) NOT NULL,
    processed_food_frequency_1_5        TINYINT NOT NULL,
    fruit_vegetable_frequency_1_5       TINYINT NOT NULL,
    legume_frequency_1_5                TINYINT NOT NULL,
    physical_activity                   VARCHAR(16) NOT NULL,
    sleep_hours                         DECIMAL(3,1) NOT NULL,
    stress_level                        VARCHAR(16) NOT NULL,
    bmi                                 DECIMAL(4,1) NOT NULL,
    pesticide_exposure                  VARCHAR(16) NOT NULL,
    air_pollution_exposure              VARCHAR(16) NOT NULL,
    occupational_chemical_exposure      VARCHAR(8) NOT NULL,
    smoking                             VARCHAR(8) NOT NULL,
    alcohol_use                         VARCHAR(16) NOT NULL,
    irregular_menstrual_cycle           VARCHAR(8) NOT NULL,
    severe_menstrual_pain               VARCHAR(8) NOT NULL,
    family_reproductive_history         VARCHAR(8) NOT NULL,
    previous_reproductive_condition     VARCHAR(8) NOT NULL,
    predicted_risk                      VARCHAR(16) NOT NULL,
    prob_low                            DECIMAL(6,5) NOT NULL,
    prob_moderate                       DECIMAL(6,5) NOT NULL,
    prob_high                           DECIMAL(6,5) NOT NULL,
    top_factor                          VARCHAR(64),
    top_factors_json                    TEXT,
    recommendations_json                TEXT,
    registered_at                       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_predicted_risk (predicted_risk),
    INDEX idx_province (province),
    INDEX idx_registered_at (registered_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
