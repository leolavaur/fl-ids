let
  rev = "6c3a1e7c25780d21033ba20d29e4921fc34508db";
  eiffel-flake = builtins.getFlake "github:phdcybersec/eiffel/${rev}";
  system = builtins.currentSystem;
  pkgs = import eiffel-flake.inputs.nixpkgs { 
    inherit system;
    config.allowUnfree = true;
  };
  eiffel = eiffel-flake.packages.${system}.eiffel-env;
in
with pkgs; mkShell {
  buildInputs = [
    # Eiffel and Python dependencies
    eiffel
    
    # LaTeX for matplotlib pgf backend
    texlive.combined.scheme-full

    # Shell environment
    getopt
    bintools
    coreutils
  ];
  shellHook = ''
    export VSCODE_PYTHON_PATH=${eiffel}/bin/python
  '' + (if stdenv.isLinux then ''
    export LD_LIBRARY_PATH=${ lib.strings.concatStringsSep ":" [
        "${cudaPackages.cudatoolkit}/lib"
        "${cudaPackages.cudatoolkit.lib}/lib"
        "${cudaPackages.cudnn}/lib"
        "${cudaPackages.cudatoolkit}/nvvm/libdevice/"
      ] }
    
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}
  '' else "") + (if builtins.pathExists ../eiffel/eiffel then ''
    export PYTHONPATH="$(realpath ../eiffel/eiffel):$PYTHONPATH";
  '' else "");
}