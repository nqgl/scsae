import sys

# sys.path.append("./nqgl/hsae_re/")
from nqgl.sc_sae.from_hsae_re.stored_acts_buffer import ActsConfig, store_acts

ac = ActsConfig(
    2,
    5,
    dtype="fp32",
    storage_dtype="fp16",
    exclude_first_acts=False,
    set_bos=True,
    model_name="gpt2",
    max_chunk_size_mb=2048,
)


def main():
    store_acts(ac)


if __name__ == "__main__":
    main()
