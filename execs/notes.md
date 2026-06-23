# cooldown=0.2

## errored out (g.174 is bad)

- attn, n=50M, gbs=16, lridx=5 (17363033.5)
- attn, n=50M, gbs=32, lridx=3,5 (17363034.3,5)
- attn, n=50M, gbs=64, lridx=3 (17363653.3)

## submitted

- attn
  - n=150M,300M gbs=16,32,64 (17363663.(0-5) - 17363668.(0-5))
    - errored out
      - 17363665.4
  - n=20M,50M gbs=64 lridx=3 errored out (17363653.3)
