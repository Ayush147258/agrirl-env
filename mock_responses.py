{
  "_comment": "Set USE_MOCK_AI=true to use these cached responses instead of live API calls. Useful for demos with no API key or limited quota.",

  "strategist": [
    {
      "priority": "balanced",
      "moisture_threshold": 38.0,
      "harvest_price_floor": 1.0,
      "fertilize_max_day": 20,
      "pest_threshold": 2.0,
      "water_reserve_pct": 0.25,
      "reasoning": "Opening strategy: maintain balanced resource allocation across all crops."
    },
    {
      "priority": "soil_health",
      "moisture_threshold": 42.0,
      "harvest_price_floor": 1.1,
      "fertilize_max_day": 18,
      "pest_threshold": 1.5,
      "water_reserve_pct": 0.28,
      "reasoning": "Soil health trending down — shifting priority to soil recovery and early pest control."
    },
    {
      "priority": "profit",
      "moisture_threshold": 32.0,
      "harvest_price_floor": 0.95,
      "fertilize_max_day": 15,
      "pest_threshold": 2.5,
      "water_reserve_pct": 0.20,
      "reasoning": "Market price elevated and mature crops present — switching to aggressive harvest mode."
    },
    {
      "priority": "survival",
      "moisture_threshold": 50.0,
      "harvest_price_floor": 1.2,
      "fertilize_max_day": 10,
      "pest_threshold": 1.8,
      "water_reserve_pct": 0.15,
      "reasoning": "Multiple crops critically dry — switching to survival mode, emergency irrigation only."
    },
    {
      "priority": "balanced",
      "moisture_threshold": 36.0,
      "harvest_price_floor": 1.0,
      "fertilize_max_day": 22,
      "pest_threshold": 2.2,
      "water_reserve_pct": 0.30,
      "reasoning": "Late-game phase: conserving water while maintaining crop health for final harvest."
    }
  ],

  "post_mortem": [
    {
      "failure_day": 12,
      "root_cause": "Insufficient irrigation during heatwave on days 10-13",
      "key_mistake": "Prioritised fertilization over irrigation during heat spike",
      "recommended_moisture_threshold": 45.0,
      "recommended_harvest_price_floor": 1.0,
      "recommended_pest_threshold": 2.0,
      "recommended_water_reserve_pct": 0.30,
      "next_run_directive": "Increase moisture_threshold to 45 during sunny or heatwave forecasts.",
      "confidence": 0.88
    },
    {
      "failure_day": 18,
      "root_cause": "Water exhausted before final harvest window",
      "key_mistake": "Over-irrigated early crops without tracking total water budget",
      "recommended_moisture_threshold": 38.0,
      "recommended_harvest_price_floor": 1.0,
      "recommended_pest_threshold": 2.0,
      "recommended_water_reserve_pct": 0.38,
      "next_run_directive": "Reserve 38% of water from Day 1; only irrigate crops below threshold 30.",
      "confidence": 0.82
    },
    {
      "failure_day": 0,
      "root_cause": "Pest infestation left untreated — yield loss on 3 crops",
      "key_mistake": "Pest threshold too high; sprayed too late when pest_level exceeded 4",
      "recommended_moisture_threshold": 35.0,
      "recommended_harvest_price_floor": 1.0,
      "recommended_pest_threshold": 1.5,
      "recommended_water_reserve_pct": 0.25,
      "next_run_directive": "Lower pest_threshold to 1.5 and treat infestations immediately when detected.",
      "confidence": 0.79
    }
  ]
}