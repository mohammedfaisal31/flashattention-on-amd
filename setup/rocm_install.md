# ROCm Installation Guide (Ubuntu 22.04)

This guide provides steps to install the ROCm stack for AMD Radeon RX 6800.

## 1. Prerequisites
Ensure your system is updated and has the necessary build tools.
```bash
sudo apt update
sudo apt upgrade
sudo apt install wget gnupg2 shellcheck
```

## 2. Kernel Support
Modern Ubuntu kernels (5.15+) generally support RDNA2. However, for ROCm, it's recommended to use the official repository to ensure the latest driver support.

## 3. Register ROCm Repository
Example for ROCm 6.0:
```bash
sudo mkdir --parents --mode=0755 /etc/apt/keyrings
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.0 jammy main" \
    | sudo tee /etc/apt/sources.list.d/rocm.list
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' \
    | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt update
```

## 4. Install Components
```bash
sudo apt install rocm-hip-sdk
```

## 5. Post-installation
Add users to the `render` and `video` groups:
```bash
sudo usermod -aG render $USER
sudo usermod -aG video $USER
```
Relog or reboot for changes to take effect.

## 6. Verify Installation
```bash
/opt/rocm/bin/rocminfo
/opt/rocm/bin/rocm-smi
```
Check if your RX 6800 (gfx1030) is detected.
