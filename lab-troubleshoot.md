# Lab Troubleshooting

This document is for commonly found problems and their solutions when using `lab`.

## `lab` troubleshooting

### Long `lab generate` on MacOS issue
If you notice `lab generate` being very slow for you, several hours or more.
Check [this discussion](https://github.com/ggerganov/llama.cpp/discussions/2182#discussioncomment-7698315)
which is suggesting to tweak the GPU limit, which MacOS assigns. By default it's
around 60%-70% of your total RAM available, which is expressed as 0:
```
$ sudo sysctl iogpu.wired_limit_mb
iogpu.wired_limit_mb = 0
```
You can set it to any number, although it's advisable to leave 4-6GB for MacOS.

On a M1 with 16GB, it was tested that `lab generate` with that limit bumped to
12GB was able to finish in less than an hour.
```
sudo sysctl iogpu.wired_limit_mb=12288
```
Once done, make sure to reset the limit back to 0, which resets it to default.

**Note:** This value will reset to defaults after machine reboot.
