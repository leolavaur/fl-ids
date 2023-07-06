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

      # tmpOverlay = (final: prev:
      #   { 
      #     pcafl = (final.poetry2nix.mkPoetryEnv {
      #       projectDir = ./exps/pcafl;
      #       editablePackageSources = { pcafl = ./exps/pcafl; };
      #       preferWheels = true;
      #       python = final.${pythonVer};
      #       #groups = [ "dev" ] ++ (if final.stdenv.isDarwin then [ "darwin" ] else [ "linux" ]);
      #       overrides = final.poetry2nix.defaultPoetryOverrides.extend (self: super: {
      #         tensorflow-io-gcs-filesystem = super.tensorflow-io-gcs-filesystem.overrideAttrs (old: {
      #           buildInputs = old.buildInputs ++ [ final.libtensorflow ];
      #         });
      #       });
      #     });
      #   }
      # );

      poetryOverlay = (final: prev:
        { poetry = (prev.poetry.override { python = final.${pythonVer}; }); }
      );

      /* Generate flake attributes for the given systems.

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

        forEachSystem :: [ system ] -> ( { ... }@pkgs -> { ... } ) -> { ... }
      */
      forEachSystem = systems: func: nixpkgs.lib.genAttrs systems (system:
        func (import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [
            p2nflake.overlay
            poetryOverlay
            # tmpOverlay
          ];
        })
      );

      /* Generate flake attributes for all systems.

        Uses `forEachSystem` to generate attributes for each "supported" system.
      
        Example:
          devShells = (forAllSystems (pkgs: { 
            default = pkgs.mkShell { 
              packages = [ pkgs.poetry ];
            };
          }));

        forAllSystems :: ( { ... }@pkgs -> { ... } ) -> { ... }
      */
      forAllSystems = func: (forEachSystem [ "x86_64-linux" "aarch64-darwin" ] func);

      pythonVer = "python310";

      expList = [
        "pcafl"
      ];

    in
    {
      devShells = forAllSystems (pkgs:
        let

          env = pkgs.buildEnv  {
              name = "env";
              paths = builtins.map (exp:
                pkgs.poetry2nix.mkPoetryEnv {
                  projectDir = ./exps/${exp};
                  editablePackageSources = { ${exp} = ./exps/${exp}; };
                  preferWheels = true;
                  python = pkgs.${pythonVer};
                  # groups = [ "dev" ] ++ (if pkgs.stdenv.isDarwin then [ "darwin" ] else [ "linux" ]);
                  overrides = pkgs.poetry2nix.defaultPoetryOverrides.extend (self: super: {
                    tensorflow-io-gcs-filesystem = super.tensorflow-io-gcs-filesystem.overrideAttrs (old: {
                      buildInputs = old.buildInputs ++ [ pkgs.libtensorflow ];
                    });
                  });
                }) expList;
            };
        in
        with pkgs; {
          default = mkShellNoCC {

            packages = [
              # this environment
              env
              # (pkgs.poetry2nix.mkPoetryEnv {
              #   projectDir = ./exps/pcafl;
              #   editablePackageSources = { pcafl = ./exps/pcafl; };
              #   preferWheels = true;
              #   python = pkgs.${pythonVer};
              #   groups = [ "dev" ] ++ (if pkgs.stdenv.isDarwin then [ "darwin" ] else [ "linux" ]);
              #   overrides = pkgs.poetry2nix.defaultPoetryOverrides.extend (self: super: {
              #     tensorflow-io-gcs-filesystem = super.tensorflow-io-gcs-filesystem.overrideAttrs (old: {
              #       buildInputs = old.buildInputs ++ [ pkgs.libtensorflow ];
              #     });
              #   });
              # })

              # other development dependencies
              # poetry
              #pcafl
            ];

            shellHook = ''
              export PYTHONPATH=$(realpath ./libs/eiffel):${ lib.strings.concatStringsSep ":" (map (p: "$(realpath ./exps/${p})") expList) }
              export EIFFEL_INTERPRETER_PATH=${env}
            '' + (if stdenv.isLinux then ''
              export LD_LIBRARY_PATH=${ lib.strings.concatStringsSep ":" [
                "${cudaPackages.cudatoolkit}/lib"
                "${cudaPackages.cudatoolkit.lib}/lib"
                "${cudaPackages.cudnn}/lib"
              ]}
            '' else "") + ":$LD_LIBRARY_PATH";
          };

        });

      packages = forAllSystems (pkgs: {
        poetry = pkgs.poetry;
      });

    };
}
