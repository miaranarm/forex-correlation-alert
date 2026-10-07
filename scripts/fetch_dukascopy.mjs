import { getHistoricalRates } from "dukascopy-node";

const args = process.argv.slice(2);
const get = (name) => args[args.indexOf(name) + 1];
const from = get("--from");
const to = get("--to");
const pairs = get("--pairs").split(",").filter(Boolean);

const results = await Promise.all(pairs.map(async (instrument) => {
  const data = await getHistoricalRates({
    instrument: instrument.toLowerCase(),
    dates: { from: new Date(from + "T00:00:00Z"), to: new Date(to + "T23:59:59Z") },
    timeframe: "m15",
    priceType: "bid",
    format: "array",
    volumes: false,
  });
  return [instrument, data.map(x => ({ timestamp: x[0], close: x[4] }))];
}));

process.stdout.write(JSON.stringify(Object.fromEntries(results)));
