{pkgs ? import (import ./npins).nixpkgs {}}: let
  python = pkgs.python314;

  app = python.pkgs.buildPythonApplication {
    pname = "gemini-websearch-mcp";
    version = "0.1.0";
    pyproject = true;
    src = ./.;

    build-system = with python.pkgs; [uv-build];

    dependencies = with python.pkgs; [
      fastmcp
      google-genai
      asyncpg
    ];

    pythonImportsCheck = ["server"];
  };

  entrypoint = pkgs.writeScriptBin "server" ''
    #!${pkgs.bash}/bin/bash
    exec ${app}/bin/python ${app}/${python.sitePackages}/server.py
  '';
in
  pkgs.dockerTools.buildImage {
    name = "gemini-websearch";
    tag = "latest";
    config = {
      Entrypoint = ["${entrypoint}/bin/server"];
    };
    copyToRoot = pkgs.buildEnv {
      name = "image-root";
      paths = [
        entrypoint
        pkgs.dockerTools.caCertificates
      ];
    };
  }
