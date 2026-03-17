{pkgs ? import (import ./npins).nixpkgs {}}: let
  sources = import ./npins;

  pyproject-nix = pkgs.callPackage sources."pyproject.nix" {};
  uv2nix = pkgs.callPackage sources.uv2nix {
    inherit pyproject-nix;
  };

  workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = ./.;};
  overlay = workspace.mkPyprojectOverlay {sourcePreference = "wheel";};

  python = pkgs.python312;

  # Wrap nixpkgs Python packages with passthru.dependencies = {} (attrset) so
  # pyproject-nix's resolveNonCyclic resolver (which calls attrNames dependencies)
  # doesn't fail on the nixpkgs list []. Uses // to avoid re-evaluating the
  # nixpkgs mkDerivation, which also reads passthru.dependencies as a list.
  wrapBuildPkg = pkg:
    pkg // {passthru = (pkg.passthru or {}) // {dependencies = {};};};

  # Provide build-system stubs that pyproject-nix's resolveBuildSystem can find.
  # These packages are only looked up by name; the actual setuptools binary is
  # injected via NIX_PYPROJECT_PYTHONPATH in pyprojectOverrides below.
  buildSystemOverlay = _final: _prev: {
    setuptools = wrapBuildPkg python.pkgs.setuptools;
    wheel = wrapBuildPkg python.pkgs.wheel;
    flit-core = wrapBuildPkg python.pkgs.flit-core;
    hatchling = wrapBuildPkg python.pkgs.hatchling;
  };

  # uv.lock doesn't carry [build-system] metadata for the workspace root package.
  # The pyproject-nix configure hook resets PYTHONPATH to just the interpreter's
  # sysconfig path, so nixpkgs packages in nativeBuildInputs are invisible.
  # Instead, inject setuptools/wheel site-packages directly into
  # NIX_PYPROJECT_PYTHONPATH (which the build hook prepends to PYTHONPATH) via preBuild.
  pyprojectOverrides = _final: prev: {
    gemini-websearch-mcp = prev.gemini-websearch-mcp.overrideAttrs (old: {
      preBuild = (old.preBuild or "") + ''
        export NIX_PYPROJECT_PYTHONPATH="${python.pkgs.setuptools}/${python.sitePackages}:${python.pkgs.wheel}/${python.sitePackages}:$NIX_PYPROJECT_PYTHONPATH"
      '';
    });
  };

  pythonSet =
    (pkgs.callPackage pyproject-nix.build.packages {inherit python;})
    .overrideScope (pkgs.lib.composeManyExtensions [buildSystemOverlay overlay pyprojectOverrides]);

  venv = pythonSet.mkVirtualEnv "gemini-websearch-env" workspace.deps.default;

  entrypoint = pkgs.writeScriptBin "server" ''
    #!${pkgs.bash}/bin/bash
    exec ${venv}/bin/python ${./server.py}
  '';
in
  pkgs.dockerTools.buildImage {
    name = "gemini-websearch";
    tag = "latest";
    config = {
      Entrypoint = ["${entrypoint}/bin/server"];
      ExposedPorts = {"8080/tcp" = {};};
    };
    copyToRoot = pkgs.buildEnv {
      name = "image-root";
      paths = [
        entrypoint
        pkgs.dockerTools.caCertificates
      ];
    };
  }
