# GitHub SSH via Clash (Fixed Steps)

## 1) Clash rules (direct for SSH)
Ensure these are present and effective:

- `ssh.github.com` routed to DIRECT
- `fake-ip-filter` includes `ssh.github.com`

## 2) Apply / reload
Run:
```
clashsub use 1
```

## 3) Verify SSH
```
ssh -T git@hauser-zhang.github.com
```
Expected: `Hi <user>! You've successfully authenticated...`

## 4) Push
```
git push
```

---

## Where these are set in your config
- Direct rule in profile: `/data1/hyzhang/clashctl/resources/profiles/1.yaml`
- Fake-IP filter in mixin: `/data1/hyzhang/clashctl/resources/mixin.yaml`

Below are the exact lines currently in those files.


## Current effective lines (copied)

### mixin.yaml (fake-ip-filter)
- L68: - "ssh.github.com"

### runtime.yaml (effective rules)
- L1086: - "ssh.github.com"

### profiles/1.yaml (github routing)
- L847: - DOMAIN-KEYWORD,github,🍃 Proxies
