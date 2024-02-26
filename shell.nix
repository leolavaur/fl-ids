let
  eiffel-flake = builtins.getFlake "github:phdcybersec/eiffel/4da0efe0bd501089b3f3355e46946d98fa10741c";
  system = builtins.currentSystem;
  pkgs = import eiffel-flake.inputs.nixpkgs { 
    inherit system;
    config.allowUnfree = true;
  };
  eiffel = eiffel-flake.packages.${system}.eiffel-env;
in
with pkgs; mkShell {
  buildInputs = [
    eiffel
    
    getopt
    bintools
    coreutils
  ];
  shellHook = ''
    export EIFFEL_PYTHON_PATH=${eiffel}/bin/python
  '' + (if stdenv.isLinux then ''
    export LD_LIBRARY_PATH=${ lib.strings.concatStringsSep ":" [
        "${cudaPackages.cudatoolkit}/lib"
        "${cudaPackages.cudatoolkit.lib}/lib"
        "${cudaPackages.cudnn}/lib"
        "${cudaPackages.cudatoolkit}/nvvm/libdevice/"
      ] }
    
    export XLA_FLAGS=--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}
  '' else "");
}