## v0.18

### Features

* Add `--backend` flag to `ilab model serve` command to allow for specifying the backend to use
   when serving a model. This is useful when you have multiple backends installed and want to
   specify which one to use. Currently, the only supported backend is `llama-cpp`.

### Breaking Changes

* `ilab model train`: The '--device' parameter no longer supports specifying a GPU index (e.g., 'cuda:0'). To use a specific GPU, set the visible GPU before running the train command.

## v0.17

### Features

#### ilab command redesign

The ilab command redesign included in v0.17 introduces a new command structure that follows a resource group design. This means that commands that once were something like `ilab chat` now are `ilab model chat`. The new groups are model, data, taxonomy, and config. The commands that fall under these are all of the pre-existing `ilab` commands just now grouped by the resource which the command commonly deals with.

The old command structure is still aliased to work but will be removed in 0.19.0. This means for 0.17.0 and 0.18.0 the aliases will exist and work as expected.

### Breaking Changes
