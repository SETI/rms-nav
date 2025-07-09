import pstats

stats = pstats.Stats('./prof/combined.prof')
# stats.print_stats()
stats.sort_stats('cumulative').print_stats() # Sort by cumulative time
