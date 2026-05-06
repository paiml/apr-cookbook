//! # Monte-Carlo Monty Hall Problem
//!
//! Sim the classic Monty Hall problem: 3 doors, prize behind one,
//! contestant picks one, host opens an empty door, contestant
//! decides whether to switch. Reports win-rate for each strategy.
//!
//! Demonstrates the **MC.92** recipe for PMAT-189 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Selvin, S. (1975) American Statistician 29; vos Savant
//!  column "Ask Marilyn" (1990); Bayesian probability classic.
//!
//! Run with: cargo run --example mc_monty_hall_problem
//!
//! Added by PMAT-189 (catalog 1324→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MontyVerdict {
    Ok {
        switch_wins: u32,
        stay_wins: u32,
        switch_rate: f64,
        stay_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(games: u32, seed: u64) -> MontyVerdict {
    if games == 0 {
        return MontyVerdict::InvalidConfig;
    }
    let mut switch_wins = 0u32;
    let mut stay_wins = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..games {
        // Prize door: 0, 1, 2.
        let prize = ((lcg(&mut rng_state) >> 32) as u32) % 3;
        // Contestant initial pick.
        let pick = ((lcg(&mut rng_state) >> 32) as u32) % 3;
        // Host opens a non-prize, non-pick door.
        let mut host_door = 99u32;
        for d in 0..3 {
            if d != prize && d != pick {
                host_door = d;
                break;
            }
        }
        // If host found no valid door, that means pick == prize,
        // and any of the other two could be opened.
        if host_door == 99 {
            // Pick == prize → host opens (pick + 1) % 3.
            host_door = (pick + 1) % 3;
        }
        // Switch door = the remaining one.
        let switch_door = 3 - pick - host_door;
        // Outcome.
        if pick == prize {
            stay_wins += 1;
        }
        if switch_door == prize {
            switch_wins += 1;
        }
    }
    MontyVerdict::Ok {
        switch_wins,
        stay_wins,
        switch_rate: f64::from(switch_wins) / f64::from(games),
        stay_rate: f64::from(stay_wins) / f64::from(games),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_monty_hall_problem")?;

    println!("typical: {:?}", simulate(10_000, 42));
    println!("invalid: {:?}", simulate(0, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn switch_rate_near_two_thirds() {
        // Classic result: P(switch wins) = 2/3.
        let v = simulate(20_000, 42);
        if let MontyVerdict::Ok { switch_rate, .. } = v {
            assert!((switch_rate - 2.0 / 3.0).abs() < 0.05);
        }
    }

    #[test]
    fn stay_rate_near_one_third() {
        // P(stay wins) = 1/3.
        let v = simulate(20_000, 42);
        if let MontyVerdict::Ok { stay_rate, .. } = v {
            assert!((stay_rate - 1.0 / 3.0).abs() < 0.05);
        }
    }

    #[test]
    fn switch_rate_higher_than_stay() {
        let v = simulate(10_000, 42);
        if let MontyVerdict::Ok {
            switch_rate,
            stay_rate,
            ..
        } = v
        {
            assert!(switch_rate > stay_rate);
        }
    }

    #[test]
    fn invalid_zero_games() {
        assert_eq!(simulate(0, 42), MontyVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 42);
        let b = simulate(500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rates_in_unit_range() {
        let v = simulate(1000, 42);
        if let MontyVerdict::Ok {
            switch_rate,
            stay_rate,
            ..
        } = v
        {
            assert!((0.0..=1.0).contains(&switch_rate));
            assert!((0.0..=1.0).contains(&stay_rate));
        }
    }

    #[test]
    fn rates_sum_to_one() {
        // Switch wins iff stay loses (and vice-versa) in this 3-door setup.
        let v = simulate(10_000, 42);
        if let MontyVerdict::Ok {
            switch_rate,
            stay_rate,
            ..
        } = v
        {
            assert!((switch_rate + stay_rate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn wins_le_games() {
        let v = simulate(1000, 42);
        if let MontyVerdict::Ok {
            switch_wins,
            stay_wins,
            ..
        } = v
        {
            assert!(switch_wins <= 1000);
            assert!(stay_wins <= 1000);
        }
    }

    #[test]
    fn single_game_works() {
        let v = simulate(1, 42);
        assert!(matches!(v, MontyVerdict::Ok { .. }));
    }

    #[test]
    fn larger_sample_more_accurate() {
        let small = simulate(100, 42);
        let big = simulate(50_000, 42);
        if let (MontyVerdict::Ok { switch_rate: s, .. }, MontyVerdict::Ok { switch_rate: b, .. }) =
            (small, big)
        {
            // Big sample should be closer to 2/3.
            let target = 2.0 / 3.0;
            assert!((b - target).abs() <= (s - target).abs() + 0.05);
        }
    }

    #[test]
    fn different_seeds_consistent() {
        let v1 = simulate(10_000, 1);
        let v2 = simulate(10_000, 2);
        if let (
            MontyVerdict::Ok {
                switch_rate: r1, ..
            },
            MontyVerdict::Ok {
                switch_rate: r2, ..
            },
        ) = (v1, v2)
        {
            // Both should be near 2/3.
            assert!((r1 - 2.0 / 3.0).abs() < 0.05);
            assert!((r2 - 2.0 / 3.0).abs() < 0.05);
        }
    }
}
