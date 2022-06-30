{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/6141b8932a5cf376fe18fcd368cecd9ad946cb68";
    utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    utils,
  }: let
    out = system: let
      useCuda = system == "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.cudaSupport = useCuda;
        config.allowUnfree = true;
      };
      inherit (pkgs) poetry2nix lib stdenv fetchurl fetchFromGitHub;
      inherit (pkgs.cudaPackages) cudatoolkit;
      inherit (pkgs.linuxPackages) nvidia_x11;
      python = pkgs.python39;
      pythonEnv = poetry2nix.mkPoetryEnv {
        inherit python;
        projectDir = ./.;
        preferWheels = true;
        overrides = poetry2nix.overrides.withDefaults (pyfinal: pyprev: rec {
          # Provide non-python dependencies.
          tokenizers = pyprev.tokenizers.overridePythonAttrs (old:
            with pkgs; {
              nativeBuildInputs =
                (old.nativeBuildInputs or [])
                ++ (with rustPlatform; [
                  rust.rustc
                  rust.cargo
                  pyfinal.setuptools-rust
                  pkg-config
                ]);
              buildInputs =
                [
                  openssl
                ]
                ++ lib.optionals stdenv.isDarwin [
                  libiconv
                  darwin.Security
                ]
                ++ lib.optionals stdenv.isLinux [
                  rustc
                ];
            });
          # Use cuda-enabled pytorch as required
          torch =
            if useCuda
            then
              # Override the nixpkgs bin version instead of
              # poetry2nix version so that rpath is set correctly.
              pyprev.pytorch-bin.overridePythonAttrs
              (old: {
                inherit (old) pname version;
                src = fetchurl {
                  url = "https://download.pytorch.org/whl/cu115/torch-1.11.0%2Bcu115-cp39-cp39-linux_x86_64.whl";
                  sha256 = "sha256-64HQZ7vP6ETJXF0n4myXqWqJNCfMRosiWerw7ZPaHH0=";
                };
              })
            else pyprev.torch;

          cffi = pyprev.cffi.overridePythonAttrs (old: {
            # Current poetry2nix override appears to be broken: `substitute(): ERROR: file 'setup.py' does not exist`.
            # https://github.com/nix-community/poetry2nix/blob/abc47c71a4920e654e7b2e4261e3e6399bbe2be6/overrides/default.nix#L260
            prePatch = null;
          });
          # Provide non-python dependencies.
          watchfiles = pyprev.watchfiles.overridePythonAttrs (old: let
            inherit (old) pname version;
            src = fetchFromGitHub {
              owner = "samuelcolvin";
              repo = "watchfiles";
              rev = "v${version}";
              sha256 = "sha256-DibxoVH7uOy9rxzhiN4HmihA7HtdzErmJOnsI/NWY5I=";
            };
          in
            with pkgs; {
              inherit src;
              cargoDeps = rustPlatform.fetchCargoTarball {
                inherit src;
                name = "${pname}-${version}";
                sha256 = "sha256-EakC/rSIS42Q4Y0pvWKG7mzppU5KjCktnC09iFMZM0A=";
              };
              buildInputs =
                (old.buildInputs or [])
                ++ lib.optionals stdenv.isDarwin [
                  libiconv
                  darwin.apple_sdk.frameworks.CoreServices
                ];
              nativeBuildInputs =
                (old.nativeBuildInputs or [])
                ++ (with rustPlatform; [
                  cargoSetupHook
                  maturinBuildHook
                ]);
            });
          orjson = pyprev.orjson.overridePythonAttrs (old: let
            inherit (old) pname version;
            src = fetchFromGitHub {
              owner = "ijl";
              repo = "orjson";
              rev = version;
              sha256 = "sha256-N0PbBU4PRqR2CAd5Ndm+vabQfaSXHBxAATwRuC+5peo=";
            };
          in
            with pkgs; {
              inherit src;
              format = "pyproject";
              cargoDeps = rustPlatform.fetchCargoTarball {
                inherit src;
                name = "${pname}-${version}";
                sha256 = "sha256-IXwz8BleoDSnf4EKZJPpHHHUxnbjUhKcIPG2QzDha54=";
              };
              buildInputs =
                (old.buildInputs or [])
                ++ lib.optionals stdenv.isDarwin [
                  libiconv
                  stdenv.cc.cc.lib
                ];
              nativeBuildInputs =
                (old.nativeBuildInputs or [])
                ++ (with rustPlatform; [
                  cargoSetupHook
                  maturinBuildHook
                ]);
            });
        });
      };
    in {
      devShell = pkgs.mkShell {
        buildInputs =
          [pythonEnv]
          ++ lib.optionals useCuda [
            nvidia_x11
            cudatoolkit
          ];
        shellHook =
          ''
            export pythonfaulthandler=1
            export pythonbreakpoint=ipdb.set_trace
            set -o allexport
            source .env
            set +o allexport
          ''
          + pkgs.lib.optionalString useCuda ''
            export CUDA_PATH=${cudatoolkit.lib}
            export LD_LIBRARY_PATH=${cudatoolkit.lib}/lib:${nvidia_x11}/lib
            export EXTRA_LDFLAGS="-l/lib -l${nvidia_x11}/lib"
            export EXTRA_CCFLAGS="-i/usr/include"
          '';
      };
    };
  in
    with utils.lib; eachSystem defaultSystems out;
}
