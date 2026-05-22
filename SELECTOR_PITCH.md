# Selector Pitch Script

## 30-second version

I built a real-data IPL selection intelligence system using official Cricsheet ball-by-ball match files.
Instead of only looking at runs or wickets, I engineered role-based T20 features like phase-wise strike rate, death-overs bowling control, dot-ball pressure, recent form, and consistency.
Then I combined them into a selector score that helps rank players and recommend a best XI.

## 60-second version

This project simulates how a cricket analyst would support IPL or T20 team selection.
I started with raw ball-by-ball IPL JSON data from Cricsheet and built a full analytics pipeline.
From that data I created player features for batters, bowlers, all-rounders, and wicketkeepers.
For batters I focused on strike rate, boundary intent, and phase-specific scoring.
For bowlers I focused on wickets, economy, dot-ball percentage, and death overs control.
Then I added recent form, consistency, and team-fit logic to build a final selector score.
The output is not just analytics tables.
It produces ranked recommendations and a suggested best XI, which makes the project more useful for actual selection discussions.

## What problem it solves

- It helps separate raw volume from real T20 impact.
- It supports shortlisting by role, not only overall totals.
- It makes squad building more data-driven.
- It creates a clear selection story for coaches and selectors.

## What to say if someone asks why this is strong

Most beginner sports analytics projects stop at descriptive charts.
This project is stronger because it turns raw cricket data into a decision-support system.
That is much closer to how analysts actually work in professional teams.
