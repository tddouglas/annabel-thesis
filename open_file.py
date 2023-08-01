import pandas as pd


# Converter used when importing Pandas data set to turn string percentage columns -> floats
def p2f(x):
    if x == '-':
        return float(0)
    elif x == '':
        print("Converting null to float")
        return float(0)
    else:
        return float(x.strip('%')) / 100


# Helper to open file and convert to Pandas DataFrame
def open_csv(filename):
    converter_mapping = {"Manganese Ore": p2f, "Copper Ore": p2f, "Nickel Ore": p2f, "Cobalt Ore": p2f, "Zinc Ore": p2f,
                         "Chromium Ore": p2f, "Molybdenum Ore": p2f, "Rare Earth Metals": p2f, "Natural Graphite": p2f,
                         "Artificial Graphite": p2f, "Lithium Oxide": p2f, "Silicon": p2f,
                         "Max Export Across Categories": p2f, "Sum Exported Across Categories": p2f,
                         "Semiconductor devices": p2f, "Electric motors": p2f, "Electric parts": p2f,
                         "Secondary batteries": p2f, "Steam turbines": p2f, "Hydraulic turbines": p2f,
                         "Gas turbines": p2f, "Electrolysers": p2f,
                         "Max Inflation Rate": p2f, "Min Government Net Lending-Borrowing Ratio": p2f,
                         "Total Exports to China": p2f, "Total Imports from China": p2f}
    datatype_mapping = {"Country Name": "string", "Swap Recipient": int,
                        "GDP Per Capita": float, "Geographic proximity to China": float, "Population Size": float}
    try:
        with open(filename, 'r', encoding='utf-8-sig') as f:
            return pd.read_csv(f, dtype=datatype_mapping, converters=converter_mapping,
                               index_col="Country Name")
    finally:
        f.close()