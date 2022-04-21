defmodule Penrose.DataTools do
  # to_tensor = fn df ->
  #   df
  #   |> Explorer.DataFrame.names()
  #   |> Enum.map(&(Explorer.Series.to_tensor(df[&1]) |> Nx.new_axis(-1)))
  #   |> Nx.concatenate(axis: 1)
  # end

  def to_tensor(data_frame) do
    data_frame
    |> Explorer.DataFrame.names()
    |> Enum.map(&(Explorer.Series.to_tensor(data_frame[&1]) |> Nx.new_axis(-1)))
    |> Nx.concatenate(axis: 1)
  end
end
