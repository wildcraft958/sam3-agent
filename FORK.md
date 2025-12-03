# Fork Information

This is a fork of the original [facebookresearch/sam3](https://github.com/facebookresearch/sam3) repository.

## Fork Details

- **Original Repository:** [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- **Fork Repository:** [wildcraft958/sam3](https://github.com/wildcraft958/sam3)
- **Purpose:** Custom modifications and improvements

## Modifications

### Agent Core (`sam3/agent/agent_core.py`)

1. **Disabled `segment_phrase` tool** (commented out in TOOLS array)
   - The `segment_phrase` tool has been temporarily disabled/commented out
   - Other segmentation tools remain active:
     - `segment_phrase_in_region`
     - `segment_phrase_with_tiling`
     - All filtering and selection tools

## Syncing with Upstream

To sync with the original repository:

```bash
# Fetch updates from upstream
git fetch upstream

# Merge upstream changes into your branch
git checkout main
git merge upstream/main

# Or rebase your changes on top of upstream
git rebase upstream/main
```

## Contributing

If you want to contribute changes back to the upstream repository, please follow the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines.

## License

This fork maintains the same license as the original repository. See [LICENSE](LICENSE) for details.

