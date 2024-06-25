## v0.18

### Features

### Breaking Changes

* `ilab`: **Deprecation of Python 3.9 support and withdrawal of Python 3.12 support** Due to changes to training requiring the usage of [GPTDolomite](https://github.com/instructlab/GPTDolomite), Python 3.9 is no longer supported and Python 3.12 support is currently withdrawn. If you are using either of these versions, you will need to start using either Python 3.10 or Python 3.11 to use this and subsequent versions of the CLI.
* `ilab model train`: The '--device' parameter no longer supports specifying a GPU index (e.g., 'cuda:0'). To use a specific GPU, set the visible GPU before running the train command.

## v0.17

### Features

#### ilab command redesign

The ilab command redesign included in v0.17 introduces a new command structure that follows a resource group design. This means that commands that once were something like `ilab chat` now are `ilab model chat`. The new groups are model, data, taxonomy, and config. The commands that fall under these are all of the pre-existing `ilab` commands just now grouped by the resource which the command commonly deals with.

The old command structure is still aliased to work but will be removed in 0.19.0. This means for 0.17.0 and 0.18.0 the aliases will exist and work as expected.

### Breaking Changes
