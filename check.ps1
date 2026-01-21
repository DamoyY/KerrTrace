$vsPath = "F:\Program Files\Microsoft Visual Studio\18\Insiders\"
$cargoPath = "C:\Users\daming\.cargo\bin\cargo.exe"
$devShellDll = Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"

Import-Module $devShellDll
Enter-VsDevShell -VsInstallPath $vsPath -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64 -no_logo"
& $cargoPath clippy --release --message-format=short