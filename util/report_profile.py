import pstats

def main():
    stats = pstats.Stats('./prof/combined.prof')
    # stats.print_stats()
    stats.sort_stats('cumulative').print_stats() # Sort by cumulative time


if __name__ == '__main__':
    main()
