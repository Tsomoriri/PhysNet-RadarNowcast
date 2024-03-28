# Meeting Minutes for PhysNet-RadarNowcast Progress Discussion

## Meeting Details

- **Date:** 2024-03-15
- **Time:** 10:00 AM - 11:30 AM
- **Location:** Virtual Meeting (Zoom)
- **Facilitator:** Stefan
- **Note Taker:** Sushen

## Attendees

- Present:
  - Sushen - Student/Researcher
  - Stefan -  Research scientist/Lead
  - IS - Module Convener
- Apologies:
  - None

## Agenda Items

1. **Review of Current Progress** - _Sushen_
   - Discussion Points:
     - Redoing input and output data configuration.
     - Adding CFL condition to the model.
     - Incorporating loss for mass conservation.
     - Considering higher resolution for the model.
   - Action Items:
     - [Sushen] Explore making `ux` and `uy` trainable, due by next meeting.

2. **Future Directions** - _Sushen_
   - Discussion Points:
     - Exploring different NAS parameters.
     - Investigating various loss functions.
     - Different methods to calculate accuracy.
     - Examining boundary conditions.
     - Simpler equations exploration for Conv LSTM radar application.
   - Action Items:
     - [Sushen] Research and implement the discussed directions, ongoing.

## Decisions Made

- Agreed to focus on making `ux` and `uy` parameters trainable for the next phase.
- Decided to explore a variety of NAS parameters and loss functions to enhance model performance.

## Action Items

- [ ] **Make `ux` and `uy` trainable:** Sushen - Due by next meeting.
- [ ] **Explore different NAS parameters and loss functions:** Sushen - Ongoing.

## Additional Notes

- Stefan and IS emphasized the importance of exploring new parameters and methods to push the boundaries of the current model.
- The possibility of adding more resolution was discussed, but it will be revisited after assessing the feasibility and computational costs.

## Next Meeting

- **Date:** To be decided.
- **Time:** To be decided.
- **Location:** Virtual Meeting (Zoom)
- **Agenda:** Review of the implemented changes and discussion on the results.
