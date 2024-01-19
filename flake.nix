{

  description = "Monorepo for my PhD thesis experiments.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";

    p2nflake = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, p2nflake, ... }:
    let

      poetryOverlay = (final: prev:
        { poetry = (prev.poetry.override { python = final.${pythonVer}; }); }
      );

      /* forEachSystem :: [ system ] -> ( { ... }@pkgs -> { ... } ) -> { ... } 
      
        Generate flake attributes for the given systems.

        Flakes outputs are a set of attributes, in which each attribute is a set of
        system-specific derivations. A path to a specific devShell, for instance, can be
        `devShells.aarch64-darwin.default`. This function generates the attributes for
        each system, and returns a set of attributes with the given name, allowing to 
        write a derivation only once.

        Example:
        
          devShells = (forEachSystem [ "x86_64-linux" "aarch64-darwin" ] (pkgs: {
            default = pkgs.mkShell { 
              packages = [ pkgs.poetry ];
            };
          }));
      */
      forEachSystem = systems: func: nixpkgs.lib.genAttrs systems (system:
        func (import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [
            p2nflake.overlay
            poetryOverlay
          ];
        })
      );

      /* forAllSystems :: ( { ... }@pkgs -> { ... } ) -> { ... } 
        
        Generate flake attributes for all systems.

        Uses `forEachSystem` to generate attributes for each "supported" system.
      
        Example:
          devShells = (forAllSystems (pkgs: { 
            default = pkgs.mkShell { 
              packages = [ pkgs.poetry ];
            };
          }));
      */
      forAllSystems = func: (forEachSystem [ "x86_64-linux" "aarch64-darwin" ] func);

      pythonVer = "python310";

      expList = [
        "demo"
        "pca"
      ];

      
    in
    {
      devShells = forAllSystems (pkgs:
        let

        eiffel = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = "${self}/libs/eiffel";
          editablePackageSources = { eiffel = "${self}/libs/eiffel/"; };
          preferWheels = true;
          python = pkgs.${pythonVer};
          overrides = pkgs.poetry2nix.defaultPoetryOverrides.extend (self: super: {
            tensorflow-io-gcs-filesystem = super.tensorflow-io-gcs-filesystem.overrideAttrs (old: {
              buildInputs = old.buildInputs ++ [ pkgs.libtensorflow ];
            });
            gpustat = super.gpustat.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools super.setuptools-scm ];
            });
            pandarallel = super.pandarallel.overrideAttrs (old: {
              buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools ];
            });
            opencensus = super.opencensus.overrideAttrs (old: {
              # See: https://github.com/DavHau/mach-nix/issues/255#issuecomment-812984772
              postInstall = ''
                rm $out/lib/python3.10/site-packages/opencensus/common/__pycache__/__init__.cpython-310.pyc
                rm $out/lib/python3.10/site-packages/opencensus/__pycache__/__init__.cpython-310.pyc
              '';
            });
          });
        };

        #   env = pkgs.buildEnv  {
        #       name = "env";
        #       paths = builtins.map (exp:
        #         pkgs.poetry2nix.mkPoetryEnv {
        #           projectDir = ./exps/${exp};
        #           editablePackageSources = { ${exp} = ./exps/${exp}; };
        #           preferWheels = true;
        #           python = pkgs.${pythonVer};
        #           overrides = pkgs.poetry2nix.defaultPoetryOverrides.extend (self: super: {
        #             tensorflow-io-gcs-filesystem = super.tensorflow-io-gcs-filesystem.overrideAttrs (old: {
        #               buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.libtensorflow ];
        #             });
        #             gpustat = super.gpustat.overrideAttrs (old: {
        #               buildInputs = (old.buildInputs or [ ]) ++ [ super.setuptools super.setuptools-scm ];
        #             });
        #             opencensus = super.opencensus.overrideAttrs (old: {
        #               # See: https://github.com/DavHau/mach-nix/issues/255#issuecomment-812984772
        #               postInstall = ''
        #                 rm $out/lib/python3.10/site-packages/opencensus/common/__pycache__/__init__.cpython-310.pyc
        #                 rm $out/lib/python3.10/site-packages/opencensus/__pycache__/__init__.cpython-310.pyc
        #               '';
        #             });
        #           });
        #         }) expList ++ [ eiffel ];

        #       ignoreCollisions = true; # 

        #     };

        in
        with pkgs; {
          default = mkShellNoCC {

            packages = [
              # this environment
              eiffel

              # tools
              poetry

            ];

            shellHook = ''
              export PATH=${eiffel}/bin:$PATH
              export PYTHONPATH=${
                lib.strings.concatStringsSep ":"
                  (map (p: "$(realpath ./exps/${p}/src)") expList)
              }:$(realpath ./libs/eiffel/)
              export EIFFEL_PYTHON_PATH=${eiffel}/bin/python
            '' + (if stdenv.isLinux then ''
              export LD_LIBRARY_PATH=${ lib.strings.concatStringsSep ":" [
                "${cudaPackages.cudatoolkit}/lib"
                "${cudaPackages.cudatoolkit.lib}/lib"
                "${cudaPackages.cudnn}/lib"
                "${pkgs.cudaPackages.cudatoolkit}/nvvm/libdevice/"
              ]}:$LD_LIBRARY_PATH
              
              export XLA_FLAGS=--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}
            '' else "");
          };

        });

      packages = forAllSystems (pkgs: {
        poetry = pkgs.poetry;

      });

    };
}
