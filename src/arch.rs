use static_assertions::assert_impl_all;



#[derive(Clone, Debug, PartialEq, Default, Copy, Eq, PartialOrd, Ord, Hash)]
pub enum ArchLayer {
    #[default]
    Dense,
    Sin,
}
assert_impl_all!(ArchLayer: Send, Sync);



#[derive(Clone, Debug, PartialEq, Default, Copy, Eq, Hash)]
pub struct ArchValue {
    pub layer: ArchLayer,
    pub n_inputs: u32,
    pub n_outputs: u32,
}
assert_impl_all!(ArchValue: Send, Sync);



#[derive(Clone, Debug, PartialEq, Default, Eq, Hash)]
pub struct Arch {
    pub values: Vec<ArchValue>,
}
assert_impl_all!(Arch: Send, Sync);

impl Arch {
    fn is_correct(&self) -> bool {
        for window in self.values.windows(2) {
            let [prev, cur] = window else { unreachable!() };

            if prev.n_outputs != cur.n_inputs || prev.layer == cur.layer {
                return false;
            }
        }

        true
    }
}

impl std::str::FromStr for Arch {
    type Err = pom::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        parser::arch()
            .parse_str(s)
            .inspect(|value| assert!(value.is_correct(), "incorrect arch"))
    }
}



pub mod parser {
    use super::*;
    use pom::utf8::*;

    pub fn number<'a>() -> Parser<'a, u32> {
        one_of("01234566789")
            .repeat(0..)
            .collect()
            .convert(str::parse::<u32>)
    }

    pub fn layer_type<'a>() -> Parser<'a, ArchLayer> {
        seq("Dense").map(|_| ArchLayer::Dense)
            | seq("Sin").map(|_| ArchLayer::Sin)
    }

    pub fn spaces<'a>() -> Parser<'a, &'a str> {
        one_of(" \n\t").repeat(0..).collect()
    }

    pub fn line<'a>() -> Parser<'a, ArchValue> {
        let parser = layer_type()
            - seq(" input shape (")
            + number()
            - seq(") output shape (")
            + number()
            - sym(')');

        parser.map(|((layer, n_inputs), n_outputs)| {
            ArchValue { layer, n_inputs, n_outputs }
        })
    }

    pub fn arch<'a>() -> Parser<'a, Arch> {
        spaces()
            * list(line(), spaces()).map(|values| Arch { values })
            - end()
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn parse_number() {
            assert_eq!(number().parse_str("00123456"), Ok(123456));
        }

        #[test]
        fn parse_arch() {
            const ARCH: &str = r"Dense input shape (3) output shape (256)
                                 Sin input shape (256) output shape (256)
                                 Dense input shape (256) output shape (256)
                                 Sin input shape (256) output shape (256)
                                 Dense input shape (256) output shape (256)
                                 Sin input shape (256) output shape (256)
                                 Dense input shape (256) output shape (256)
                                 Sin input shape (256) output shape (256)
                                 Dense input shape (256) output shape (256)
                                 Sin input shape (256) output shape (256)
                                 Dense input shape (256) output shape (1)";

            assert_eq!(ARCH.parse::<Arch>().unwrap(), Arch { values: vec![
                ArchValue { layer: ArchLayer::Dense, n_inputs: 3, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Sin, n_inputs: 256, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Dense, n_inputs: 256, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Sin, n_inputs: 256, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Dense, n_inputs: 256, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Sin, n_inputs: 256, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Dense, n_inputs: 256, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Sin, n_inputs: 256, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Dense, n_inputs: 256, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Sin, n_inputs: 256, n_outputs: 256 },
                ArchValue { layer: ArchLayer::Dense, n_inputs: 256, n_outputs: 1 },
            ]});
        }

        #[test]
        #[should_panic]
        fn unparse() {
            const ARCH: &str = r"Dense input shape (42) output shape (69)";

            println!("{:?}", ARCH.parse::<Arch>().unwrap());
        }
    }
}