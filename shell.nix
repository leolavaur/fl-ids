let
  rev = "d7dbb474307fd82763fe2c3f74b7edeaf8338dd3";
  eiffel-flake = builtins.getFlake "github:phdcybersec/eiffel/${rev}";
  system = builtins.currentSystem;
  pkgs = import eiffel-flake.inputs.nixpkgs { 
    inherit system;
    config.allowUnfree = true;
  };
  eiffel = eiffel-flake.packages.${system}.eiffel-env;
in
if 
  pkgs.stdenv.isLinux 
then
  (eiffel-flake.devShells.${system}.default.overrideAttrs (oldAttrs: {
    shellHook = oldAttrs.shellHook + (if builtins.pathExists ../eiffel/eiffel then ''
      export PYTHONPATH="$(realpath ../eiffel/):$PYTHONPATH"
    '' else "");
  }))
else
  with pkgs; 
  mkShell {
  
    buildInputs = [
      # Eiffel and Python dependencies
      (python3.withPackages (ps: with ps; [
        numpy
        pandas
        scipy
        scikit-learn
        matplotlib
        seaborn
        ipykernel
        requests
        schema
        # # For Eiffel specifically
        # keras
        # omegaconf
        # absl-py
        # optree
        # tensorflow
      ]))

      poetry
      
      # LaTeX for matplotlib pgf backend
      texliveFull

      # Shell environment
      getopt
      bintools
      coreutils
    ];

    shellHook = ''
      export PYTHONPATH="$(realpath .):$PYTHONPATH"
    '' + (if builtins.pathExists ../eiffel/eiffel then ''
      export PYTHONPATH="$(realpath ../eiffel/):$PYTHONPATH"
    '' else "");
  }