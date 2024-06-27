let
  rev = "0fdddf5a588780a95337481a3d2228aaf7f8cfec";
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
  (eiffel-flake.devShells.default.overrideAttrs (oldAttrs: {
    shellHook = oldAttrs.shellHook + (if builtins.pathExists ../eiffel/eiffel then ''
      export PYTHONPATH="$(realpath ../eiffel/):$PYTHONPATH";
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
        ipython
      ]))
      
      # LaTeX for matplotlib pgf backend
      texliveFull

      # Shell environment
      getopt
      bintools
      coreutils
    ];
  }