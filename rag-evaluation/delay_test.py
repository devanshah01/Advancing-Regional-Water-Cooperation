from incentives import IncentiveConfig, delay_factor, beats_no_deal

proposal = {"Kazakhstan": 10, "Uzbekistan": 10, "Turkmenistan": 10,
            "Tajikistan": 10, "Kyrgyzstan": 10}
no_deal = {"Kazakhstan": 6, "Uzbekistan": 6, "Turkmenistan": 6,
           "Tajikistan": 6, "Kyrgyzstan": 6}

for rate in [0.0, 0.05, 0.10, 0.20]:
    cfg = IncentiveConfig(delay_rate=rate, nodeal_decay=rate * 1.5)
    print(f"\nDelay rate {rate:.0%}")
    for rnd in range(7):
        result = beats_no_deal(proposal, no_deal, rnd, cfg)
        print(f"  Round {rnd}: {sum(result.values())}/5 countries prefer the deal")
        