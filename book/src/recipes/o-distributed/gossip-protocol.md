# Gossip Protocol

Gossip-based protocol where nodes exchange and average model parameters without a central coordinator. Each round, every node picks a random peer, and the pair averages their parameters until all nodes converge to the global average.

## CLI Equivalent
```bash
N/A
```

## Key Concepts
- Decentralized parameter averaging without a coordinator
- Random peer selection and pairwise averaging
- Convergence measurement via divergence metrics

## Run
```bash
cargo run --example distributed_gossip_protocol
```

## Source
[`examples/distributed/distributed_gossip_protocol/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/distributed/distributed_gossip_protocol/main.rs)
