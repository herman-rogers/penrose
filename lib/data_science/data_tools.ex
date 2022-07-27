defmodule Penrose.DataTools do
  def to_tensor(data_frame) do
    data_frame
    |> Explorer.DataFrame.names()
    |> Enum.map(
      &(Explorer.Series.to_tensor(data_frame[&1])
        |> Nx.new_axis(-1))
    )
    |> Nx.concatenate(axis: 1)
  end
end
