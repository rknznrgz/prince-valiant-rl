# Artifact and verification records

Machine-readable records supporting the repository documentation:

| File | Purpose |
| --- | --- |
| `artifacts.json` | Model/data/state identities, sizes, destinations, and access conditions |
| `inventory.json` | Current exported file sizes and SHA-256 hashes; excludes itself and Git metadata |
| `provenance.json` | Original/exported hashes and retired documentation records |
| `dependency_graph.json` | Import closure and dynamic dependencies |
| `source_symbols.json` | Exported source files and top-level symbols |
| `verification.json` | Preparation-time static, help, model-load, and metadata checks |
| `log_provenance.json` | Source identity for extracted Level 1 training metrics |
| `link_audit.json` | Current local documentation-link checks |

Publication and documentation reviews are separate records. Preparation checks do not establish a fresh-machine install or a new gameplay evaluation. See [reproducibility](../docs/REPRODUCIBILITY.md).
