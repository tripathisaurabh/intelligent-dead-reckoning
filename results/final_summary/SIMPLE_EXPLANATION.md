# Simple Explanation for the Team and Judges

## What did we build?

We built a system that estimates where a vehicle continues moving when GPS temporarily disappears.
It reads the phone's movement sensors, corrects their errors, applies vehicle-motion rules, and
uses the road map to keep the predicted route near a real road.

## What does the final picture show?

- **Black line:** where the vehicle actually travelled, measured by the reference equipment.
- **Red dashed line:** what basic phone-sensor calculation predicted without our complete system.
- **Green line:** what our complete system predicted.
- **Blue dot:** the position where GPS was switched off.
- **Stars:** the actual and estimated final positions after 20 seconds.

The vehicle travelled 141.995 metres. Basic sensor integration ended 136.73 metres away from the
correct position. The complete system ended 15.83 metres away.

## Simple meanings of the technical terms

| Term | Simple meaning |
|---|---|
| IMU | The phone's accelerometer and gyroscope - sensors that measure movement and rotation |
| Accelerometer | Measures how the phone/vehicle speeds up, slows down, or changes direction |
| Gyroscope | Measures how the phone/vehicle rotates or turns |
| Bias | A small sensor error that remains even when the real movement is zero |
| Drift | Position error that grows as sensor measurements are repeatedly added over time |
| Calibration | Measuring and removing initial sensor errors |
| Phone mounting alignment | Determining how the phone is rotated relative to the vehicle |
| Kalman filter / IEKF | A mathematical estimator that tracks motion and uncertainty while correcting noisy measurements |
| NHC / vehicle constraint | A rule that a normal vehicle does not slide sideways or fly upward |
| CNN | A small trained neural network that recognizes IMU patterns and adjusts sensor/filter corrections |
| GNSS | Satellite positioning systems such as GPS, Galileo and NavIC |
| GNSS blackout | A period when satellite location is unavailable or deliberately disabled for testing |
| Map matching | Comparing the estimated movement with real roads and selecting the most likely road path |
| HMM/Viterbi | The method used to select a consistent road sequence over time instead of snapping every point independently |
| Endpoint error | Distance between the predicted final position and the actual final position |
| RMSE | A score representing error throughout the complete trajectory, with larger mistakes penalized more |
| Drift percentage | Final position error divided by the distance travelled, multiplied by 100 |

## Thirty-second explanation

> We simulated a 20-second GPS outage while the vehicle travelled about 142 metres. Directly using
> the phone sensors produced 136.73 metres of final error. We then added sensor calibration, phone
> mounting correction, an invariant Kalman filter, vehicle-motion constraints, a small trained CNN,
> and road-map matching. Together these reduced final error to 15.83 metres, or 11.146% drift. This
> is an 88.4% reduction in endpoint error on the Vta08 test. Our different-driver test also showed
> that more multi-driver training is required before claiming general accuracy.

## Important honest statement

The 11.146% result belongs specifically to the controlled Vta08 blackout. It is not an accuracy
claim for every driver, phone, route or vehicle. The S3c different-driver test performed poorly,
which identifies cross-driver speed and scale generalization as the next major engineering problem.
