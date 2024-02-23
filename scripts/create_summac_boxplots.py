from matplotlib import pyplot as plt
import pandas as pd

if __name__ == "__main__":
    summac_df = pd.read_csv("outputs/xsum__tagged_examples__summac.csv")

    df = summac_df[["summac_zs", "summac_conv"]]
    rescaled_df = (df - df.mean()) / df.std()
    rescaled_df = rescaled_df.rename(columns={"summac_zs": "summac_zs_rescaled", "summac_conv": "summac_conv_rescaled"})

    concat_df = pd.concat([summac_df, rescaled_df], axis=1)

    summac_conv_df = pd.DataFrame(
        dict([(index, group_df["summac_conv_rescaled"]) for index, group_df in concat_df.groupby("label")]))
    summac_conv_df.boxplot()
    plt.savefig("figures/xsum__summac_conv_rescaled__boxplot.png")
    plt.close()

    summac_zs_df = pd.DataFrame(
        dict([(index, group_df["summac_zs_rescaled"]) for index, group_df in concat_df.groupby("label")]))
    summac_zs_df.boxplot()
    plt.savefig("figures/xsum__summac_zs_rescaled__boxplot.png")
    plt.close()