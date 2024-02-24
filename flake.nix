{

  description = "Monorepo for my PhD thesis experiments.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-22.11";

    p2nflake = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    eiffel = {
      path = "./libs/eiffel";
    }
  };

  outputs = { nixpkgs, p2nflake, eiffel, ... }:
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
        "assessment"
      ];

      
    in
    {
      devShells = forAllSystems (pkgs:
        let

        eiffelEnv = pkgs.poetry2nix.mkPoetryEnv ({
          projectDir = "${eiffel}";
          editablePackageSources = { eiffel = "${eiffel}/"; };
          preferWheels = true;
          python = pkgs.${pythonVer};
          overrides = pkgs.poetry2nix.defaultPoetryOverrides.extend (self: super: {
            # tensorflow = super.tensorflow.overrideAttrs (old: with pkgs;{
            #   propagatedBuildInputs = (old.buildInputs or [ ]) ++ [ 
            #     cudaPackages.cudatoolkit
            #     cudaPackages.cudnn
            #   ];
            #   buildInputs = (old.buildInputs or [ ]) ++ [ 
            #     cudaPackages.cudatoolkit
            #     cudaPackages.cudnn
            #   ];
            #   nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ 
            #     addOpenGLRunpath
            #   ];
            #   postFixup = ''
            #     find $out -type f \( -name '*.so' -or -name '*.so.*' \) | while read lib; do
            #       addOpenGLRunpath "$lib"

            #       patchelf --set-rpath "${cudatoolkit}/lib:${cudatoolkit.lib}/lib:${cudaPackages.cudnn}/lib:${cudaPackages.nccl}/lib:$(patchelf --print-rpath "$lib")" "$lib"
            #     done
            #   '';
            #   # XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}";
            # });
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
            # threadpoolctl = super.threadpoolctl.overrideAttrs (old: {
            #   buildInputs = (old.buildInputs or [ ]) ++ [ super.stdenv.cc.libc ];
            # });
            
          });          
        } // (if pkgs.stdenv.isDarwin then {
          pyproject = "${eiffel}/macos/pyproject.toml";
          poetrylock = "${eiffel}/macos/poetry.lock";
        } else {}));

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
              # (eiffel.env.overrideAttrs (old: {
              #   shellHook = with pkgs; ''
              #     export LD_LIBRARY_PATH=${
              #       lib.strings.concatStringsSep ":" [
              #         "${cudaPackages.cudatoolkit}/lib"
              #         "${cudaPackages.cudatoolkit.lib}/lib"
              #         "${cudaPackages.cudnn}/lib"
              #         "${cudaPackages.cudatoolkit}/nvvm/libdevice/"
              #       ]
              #     }:$LD_LIBRARY_PATH
              #     export XLA_FLAGS=--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}
              #   '';
              # }))

              # tools
              poetry
              getopt # for remoterun.sh

              bintools # ld and friends
            ];  

            shellHook = let 
              cuda = cudaPackages.cudatoolkit.overrideAttrs (old: {
                postInstall = ''
                  ln -s $out/lib64/stubs/libcuda.so $out/lib64/stubs/libcuda.so.1
                '';
              });
            in ''
              export PYTHONPATH=$(realpath ./libs/eiffel/)
              export EIFFEL_PYTHON_PATH=${eiffel}/bin/python
            '' + (if stdenv.isLinux then ''
              export LD_LIBRARY_PATH=${ lib.strings.concatStringsSep ":" [
                  "${cuda}/lib/stubs"
                  "${cudaPackages.cudatoolkit.lib}/lib"
                  "${cudaPackages.cudnn}/lib"
                  "${cudaPackages.cudatoolkit}/nvvm/libdevice/"
                ] }
              
              export XLA_FLAGS=--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}
            '' else "");
          };

        });

      packages = forAllSystems (pkgs: {
        poetry = pkgs.poetry;

      });

    };
}
