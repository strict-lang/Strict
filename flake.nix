{
  description = "Strict compiler";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      llvm = pkgs.fetchurl {
        url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.3/LLVM-23.1.3-Linux-X64.tar.xz";
        hash = "sha256-yp/jhaXZ4tiBMtwOI6AOn4fzj6dfZyB7XkV9II5x0BM=";
      };
      libraries = with pkgs; [ stdenv.cc.cc.lib zlib ncurses libxml2 ];
    in {
      packages.${system} = rec {
      default = strict;
      strict = pkgs.stdenv.mkDerivation {
        pname = "strict";
        version = "1.0.0";
        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            let base = baseNameOf path;
            in base != "Tools" && base != "obj" && base != ".git" && !(pkgs.lib.hasSuffix ".exe" base);
        };
        nativeBuildInputs = [ pkgs.xz pkgs.patchelf ];
        buildInputs = libraries ++ [ pkgs.gcc.cc pkgs.glibc ];
        libPath = pkgs.lib.makeLibraryPath libraries;
        dontStrip = true;
        dontPatchELF = true;
        buildPhase = ''
          mkdir -p Tools
          tar -xJf ${llvm} -C Tools
          bin="$PWD/Tools/LLVM-23.1.3-Linux-X64/bin"
          # The upstream LLVM binaries ask the kernel for /lib64/ld-linux-x86-64.so.2. A Nix sandbox
          # does not have the host /lib64, so exec fails with "required file not found" before any
          # library search. Point them at Nix's loader; the clang wrapper below still links outputs.
          interpreter=$(cat $NIX_CC/nix-support/dynamic-linker)
          for binary in "$bin"/*; do
            if [ -f "$binary" ] && [ ! -L "$binary" ] && patchelf --print-interpreter "$binary" >/dev/null 2>&1; then
              patchelf --set-interpreter "$interpreter" --set-rpath "$libPath:${pkgs.glibc}/lib" "$binary"
            fi
          done
          gcclib=$(dirname ${pkgs.gcc.cc}/lib/gcc/*/*/crtbegin.o)
          rm -f "$bin/clang"
          cat > "$bin/clang" << EOF
          #!/bin/sh
          link=yes
          for argument in "\$@"; do
            case \$argument in
              -c|-E|-S|-M|-MM) link=no ;;
            esac
          done
          headers="-isystem ${pkgs.glibc.dev}/include -isystem ${pkgs.linuxHeaders}/include"
          toolchain="--gcc-toolchain=${pkgs.gcc.cc} -B${pkgs.glibc}/lib -B$gcclib -B${pkgs.stdenv.cc.cc.lib}/lib"
          if [ "\$link" = yes ]; then
            exec "\$(dirname "\$0")/clang-23" \$toolchain -fuse-ld=lld \\
              -L${pkgs.glibc}/lib -L$gcclib -L${pkgs.stdenv.cc.cc.lib}/lib \\
              -Wl,-dynamic-linker,${pkgs.glibc}/lib/ld-linux-x86-64.so.2 \\
              -Wl,-rpath,${pkgs.glibc}/lib -Wl,-rpath,${pkgs.stdenv.cc.cc.lib}/lib \\
              "\$@"
          fi
          exec "\$(dirname "\$0")/clang-23" \$toolchain \$headers "\$@"
          EOF
          chmod +x "$bin/clang"
          export PATH="$bin:$PATH"
          export LD_LIBRARY_PATH="$libPath''${LD_LIBRARY_PATH:+:}$LD_LIBRARY_PATH"
          parent=$(dirname "$(pwd)")
          cd "$parent"
          mv source Strict
          cd Strict
          sh Compiler/bootstrap.sh
          cd "$parent"
          mv Strict source
        '';
        installPhase = ''
          if [ ! -f strict ]; then cd source; fi
          mkdir -p $out/bin $out/lib/strict
          cp -a . $out/lib/strict/
          interpreter=$(cat $NIX_CC/nix-support/dynamic-linker)
          tools=$out/lib/strict/Tools/LLVM-23.1.3-Linux-X64/bin
          chmod -R u+w $out/lib/strict/strict "$tools"
          for binary in $out/lib/strict/strict "$tools"/*; do
            if [ -f "$binary" ]; then
              patchelf --set-interpreter "$interpreter" --set-rpath "$libPath" "$binary" || true
            fi
          done
          printf '%s\n' '#!/bin/sh' "export PATH=$tools:\$PATH" "exec $out/lib/strict/strict \"\$@\"" > $out/bin/strict
          chmod +x $out/bin/strict
        '';
      };
      };
    };
}
