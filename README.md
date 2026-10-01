# Experimental Data

The raw tractor CAN-bus data used by the analysis are not included in this repository.

The analysis expects field-level files following the naming convention:

```text
FIELD_*_RAW.tab
```

For example:

```text
FIELD_1_RAW.tab
FIELD_2_RAW.tab
FIELD_3_RAW.tab
```

The following variables are used by the analysis:

| Variable                  | Description                  |
| ------------------------- | ---------------------------- |
| `EngineSpeed`             | Engine speed                 |
| `ActualEngine_PercTorque` | Indicated engine torque/load |
| `EngFuelRate`             | Engine fuel rate             |
| `WheelBasedVehicleSpeed`  | Wheel-based vehicle speed    |
| `HtcPositionSensPerc`     | Hitch position               |
| `HtcDraftSens1Perc`       | Hitch draft sensor 1         |
| `HtcDraftSens2Perc`       | Hitch draft sensor 2         |
| `PTO_Status`              | PTO status                   |
| `KeypadButton`            | Implement/keypad information |

The files should be tab-separated text files.

Because the experimental data may be subject to data-sharing restrictions, users should obtain the appropriate permission before redistributing or publishing the raw measurements.

If a public data repository is associated with the research article, its DOI should be added here.
