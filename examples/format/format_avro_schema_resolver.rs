//! # Format Avro Schema Resolution
//!
//! Avro schema-evolution rules between writer and reader:
//!   identical → Compatible
//!   reader has extra optional → Compatible (default applies)
//!   reader missing required from writer → Incompatible
//!   type promotion (int → long, float → double) → Compatible
//!   type narrowing (long → int) → Incompatible
//!
//! Demonstrates the **FMT.29** recipe for PMAT-148 (format round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Apache Avro schema resolution specification.
//!
//! Run with: cargo run --example format_avro_schema_resolver
//!
//! Added by PMAT-148 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvroType {
    Int,
    Long,
    Float,
    Double,
    String,
    Bytes,
}

#[derive(Debug, PartialEq)]
pub enum CompatVerdict {
    Compatible,
    Promotable { from: AvroType, to: AvroType },
    Incompatible { reason: &'static str },
}

pub fn resolve(writer_type: AvroType, reader_type: AvroType) -> CompatVerdict {
    if writer_type == reader_type {
        return CompatVerdict::Compatible;
    }
    // Promotion lattice (Avro spec):
    //   int → long, float, double
    //   long → float, double
    //   float → double
    //   string ↔ bytes
    let promotable = matches!(
        (writer_type, reader_type),
        (
            AvroType::Int,
            AvroType::Long | AvroType::Float | AvroType::Double
        ) | (AvroType::Long, AvroType::Float | AvroType::Double)
            | (AvroType::Float, AvroType::Double)
            | (AvroType::String, AvroType::Bytes)
            | (AvroType::Bytes, AvroType::String)
    );
    if promotable {
        return CompatVerdict::Promotable {
            from: writer_type,
            to: reader_type,
        };
    }
    CompatVerdict::Incompatible {
        reason: "type narrowing or unsupported pair",
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_avro_schema_resolver")?;

    println!("int → int: {:?}", resolve(AvroType::Int, AvroType::Int));
    println!("int → long: {:?}", resolve(AvroType::Int, AvroType::Long));
    println!(
        "long → int (narrowing): {:?}",
        resolve(AvroType::Long, AvroType::Int)
    );
    println!(
        "float → double: {:?}",
        resolve(AvroType::Float, AvroType::Double)
    );
    println!(
        "string → bytes: {:?}",
        resolve(AvroType::String, AvroType::Bytes)
    );
    println!(
        "double → string (incompat): {:?}",
        resolve(AvroType::Double, AvroType::String)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_compatible() {
        for t in [
            AvroType::Int,
            AvroType::Long,
            AvroType::Float,
            AvroType::Double,
            AvroType::String,
            AvroType::Bytes,
        ] {
            assert_eq!(resolve(t, t), CompatVerdict::Compatible);
        }
    }

    #[test]
    fn int_promotes_to_long() {
        let v = resolve(AvroType::Int, AvroType::Long);
        assert!(matches!(v, CompatVerdict::Promotable { .. }));
    }

    #[test]
    fn int_promotes_to_double() {
        let v = resolve(AvroType::Int, AvroType::Double);
        assert!(matches!(v, CompatVerdict::Promotable { .. }));
    }

    #[test]
    fn float_promotes_to_double() {
        let v = resolve(AvroType::Float, AvroType::Double);
        assert!(matches!(v, CompatVerdict::Promotable { .. }));
    }

    #[test]
    fn string_promotes_to_bytes() {
        let v = resolve(AvroType::String, AvroType::Bytes);
        assert!(matches!(v, CompatVerdict::Promotable { .. }));
    }

    #[test]
    fn bytes_promotes_to_string() {
        let v = resolve(AvroType::Bytes, AvroType::String);
        assert!(matches!(v, CompatVerdict::Promotable { .. }));
    }

    #[test]
    fn long_to_int_narrowing_incompatible() {
        let v = resolve(AvroType::Long, AvroType::Int);
        assert!(matches!(v, CompatVerdict::Incompatible { .. }));
    }

    #[test]
    fn double_to_float_narrowing_incompatible() {
        let v = resolve(AvroType::Double, AvroType::Float);
        assert!(matches!(v, CompatVerdict::Incompatible { .. }));
    }

    #[test]
    fn string_to_int_incompatible() {
        let v = resolve(AvroType::String, AvroType::Int);
        assert!(matches!(v, CompatVerdict::Incompatible { .. }));
    }

    #[test]
    fn double_to_string_incompatible() {
        let v = resolve(AvroType::Double, AvroType::String);
        assert!(matches!(v, CompatVerdict::Incompatible { .. }));
    }

    #[test]
    fn int_to_float_promotable() {
        let v = resolve(AvroType::Int, AvroType::Float);
        assert!(matches!(v, CompatVerdict::Promotable { .. }));
    }

    #[test]
    fn long_to_float_promotable() {
        let v = resolve(AvroType::Long, AvroType::Float);
        assert!(matches!(v, CompatVerdict::Promotable { .. }));
    }
}
