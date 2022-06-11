#!/usr/bin/env scriptisto

// scriptisto-begin
// script_src: src/main.rs
// build_cmd: cargo build --release && strip ./target/release/script
// target_bin: ./target/release/script
// files:
//  - path: Cargo.toml
//    content: |
//     package = { name = "script", version = "0.1.0", edition = "2018"}
//     [dependencies]
//     clap = { version = "3.1.18", features = ["derive"] }
// scriptisto-end

use std::fs;
use clap::Parser;

fn fix(line: &str) -> String {
    line.replace(", )", ")")
        .replace(", ]", "]")
        .replace(", }", "}")
}

fn splitlines(line: &str) -> String {
    let delims = "]})";

    let mut out_line = String::new();
    for c in line.chars() {
        out_line.push(c);
        if delims.find(c).is_some() {
            out_line += "\n";
        }
    }

    out_line
}

fn parse_metis(log: &str) -> (String, String) {
    enum State {
        PreMetis,
        Metis,
    }

    let mut metis_log = String::new();
    let mut extras = String::new();

    let mut state = State::PreMetis;

    for line in log.lines() {
        let mut target: &mut String = &mut extras;

        match state {
            State::PreMetis => {
                if line.starts_with("     Running `") {
                    state = State::Metis;
                }
            }
            State::Metis => {
                target = &mut metis_log;
            }
        }

        target.to_owned().push_str(&splitlines(&fix(line)));
        target.push('\n');
    }

    (metis_log, extras)
}

fn parse_smite(log: &str) -> (String, String) {
    enum State {
        PreSmite,
        Smite,
    }

    let mut smite_log = String::new();
    let mut extras = String::new();

    let mut state = State::PreSmite;

    for line in log.lines() {
        let mut target: &mut String = &mut extras;

        match state {
            State::PreSmite => {
                if line.starts_with("METIS RESULT: [") {
                    state = State::Smite;
                }
            }
            State::Smite => {
                target = &mut smite_log;
            }
        }

        target.to_owned().push_str(&splitlines(&fix(line)));
        target.push('\n');
    }

    (smite_log, extras)
}

#[derive(Parser, Debug)]
#[clap(name = "script", about = "A script.")]
struct Args {
    metis_log_path: String,
    smite_log_path: String,
    extras_log_path: String,
}

fn main() -> std::io::Result<()> {
    let opt = Args::from_args();

    let metis_log = fs::read_to_string(&opt.metis_log_path)?;
    let smite_log = fs::read_to_string(&opt.smite_log_path)?;

    let (metis_log, metis_extras) = parse_metis(&metis_log);
    let (smite_log, smite_extras) = parse_smite(&smite_log);

    fs::write(&opt.metis_log_path, metis_log)?;
    fs::write(&opt.smite_log_path, smite_log)?;
    fs::write(&opt.extras_log_path, metis_extras + &smite_extras)?;

    Ok(())
}
