# Use #[cfg(feature = "...")] to create a minimal binary

Here I want to opt-in to the Rust machinery to collect and validate instrumentation. When we built this and when we test this we need to have all kinds of tracing and internal measurement to validate things. I want a more production focused build by excluding things.
