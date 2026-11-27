#!/bin/sh
# Fake apr for the argv-surface falsifier: --version, and `<verb> --help` lists four flags.
case $1 in
  --version) echo "apr 0.0.0-fixture" ;;
  *) echo "  --input  --output  --verbose  --thinking" ;;
esac
