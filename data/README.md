# data/

Git ignores every file in this folder except this README. Do not commit the survey file.
The offline demo and the tests do not need this folder: `fedsugar synth --out data/synthetic.csv`
writes synthetic rows with the same columns, ranges and class shares.

## Expected file

`diabetes_012_health_indicators_BRFSS2015.csv` (the default path, or set `FEDSUGAR_DATA`).

| Column | Values | Kind |
|---|---|---|
| `Diabetes_012` (target) | 0 = no diabetes, 1 = prediabetes, 2 = diabetes | class |
| `HighBP`, `HighChol`, `CholCheck`, `Smoker`, `Stroke`, `HeartDiseaseorAttack`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `AnyHealthcare`, `NoDocbcCost`, `DiffWalk`, `Sex` | 0 or 1 | binary |
| `BMI` | 12 to 98 | numeric |
| `GenHlth` | 1 (excellent) to 5 (poor) | ordinal |
| `MentHlth`, `PhysHlth` | 0 to 30 days | numeric |
| `Age` | 1 to 13 (5-year groups) | ordinal |
| `Education` | 1 to 6 | ordinal |
| `Income` | 1 to 8 | ordinal |

The loader stops if a column is missing, a value is not numeric or a value is outside its range.

## Source

| Item | Value |
|---|---|
| Name | Diabetes Health Indicators Dataset (a cleaned subset of the CDC BRFSS 2015 telephone survey) |
| URL | https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset |
| Original survey | CDC Behavioral Risk Factor Surveillance System, https://www.cdc.gov/brfss/ |
| License | CC0 (public domain) on Kaggle. The survey is public, de-identified data |
| Size | 253,680 rows, 22 columns |

## How to download

1. Download `diabetes_012_health_indicators_BRFSS2015.csv` from the Kaggle page.
2. Save it as `data/diabetes_012_health_indicators_BRFSS2015.csv`.
3. Run `fedsugar partition` to check the file and to see the client split.
