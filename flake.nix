{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        devEnv = (pkgs.buildFHSUserEnv {
        name = "instuctlab";
        targetPkgs = pkgs: (with pkgs; [
          llama-cpp
          micromamba
          # llama-cpp-python requirements
          cmake
          ninja
          gcc
          gh
        ]);
        runScript = "zsh";

        profile = ''
          eval "$(micromamba shell hook -s posix)"
          export MAMBA_ROOT_PREFIX=.mamba
          micromamba create -q -n instructlab
          micromamba activate instructlab
          micromamba install --yes -f env.yaml
          pip install -e .
        '';


      }).env;
    in {
      devShell = devEnv;
    }
  );
}