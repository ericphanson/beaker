// beaker-stamp-derive/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Expr, Fields, Lit, Meta};

#[proc_macro_derive(Stamp, attributes(stamp))]
pub fn derive_stamp(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, data, .. } = parse_macro_input!(input);

    let mut entries = Vec::new();

    let fields = match data {
        Data::Struct(s) => match s.fields {
            Fields::Named(named) => named.named,
            _ => {
                return quote! { compile_error!("Stamp derive supports named fields only"); }.into()
            }
        },
        _ => return quote! { compile_error!("Stamp derive supports structs only"); }.into(),
    };

    for f in fields {
        let name = f.ident.unwrap();
        let mut include = false;
        let mut rename: Option<String> = None;
        let mut with_fn: Option<syn::Path> = None;

        for attr in f.attrs.iter().filter(|a| a.path().is_ident("stamp")) {
            if attr.meta.path().is_ident("stamp") {
                match &attr.meta {
                    Meta::Path(_) => {
                        include = true; // #[stamp]
                    }
                    Meta::List(_list) => {
                        include = true; // #[stamp(...)]
                        let _result = attr.parse_nested_meta(|meta| {
                            if meta.path.is_ident("rename") {
                                let value = meta.value()?;
                                if let Expr::Lit(expr_lit) = value.parse()? {
                                    if let Lit::Str(s) = expr_lit.lit {
                                        rename = Some(s.value());
                                    }
                                }
                            } else if meta.path.is_ident("with") {
                                let value = meta.value()?;
                                if let Expr::Lit(expr_lit) = value.parse()? {
                                    if let Lit::Str(s) = expr_lit.lit {
                                        with_fn =
                                            Some(syn::parse_str(&s.value()).expect("valid path"));
                                    }
                                }
                            }
                            Ok(())
                        });
                    }
                    _ => {}
                }
            }
        }

        if include {
            let key_lit = rename.clone().unwrap_or_else(|| name.to_string());
            let value_expr = if let Some(fun) = with_fn.clone() {
                quote! { #fun(&self.#name) }
            } else {
                quote! { &self.#name }
            };

            entries.push(quote! {
                map.insert(#key_lit.to_string(), serde_json::to_value(#value_expr).unwrap());
            });
        }
    }

    let gen = quote! {
        impl ::beaker_stamp::Stamp for #ident {
            fn stamp_value(&self) -> serde_json::Value {
                let mut map = serde_json::Map::new();
                #(#entries)*
                serde_json::Value::Object(map)
            }
        }
    };
    gen.into()
}
