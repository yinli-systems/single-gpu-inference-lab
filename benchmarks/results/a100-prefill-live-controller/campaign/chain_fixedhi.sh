#!/bin/bash
until grep -q "A100_LIVE_DONE\|RUN_FAILED\|busy\|DIRTY" /root/lab/campaign_a100_live.log; do sleep 20; done
grep -q "A100_LIVE_DONE" /root/lab/campaign_a100_live.log || { echo "main block did not finish cleanly - not starting"; exit 1; }
sleep 10
exec /root/lab/campaign_a100_live_fixedhi.sh
