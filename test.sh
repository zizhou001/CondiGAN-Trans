#!/bin/bash
echo "Starting Python script" > /root/autodl-tmp/project/output.log
nohup python -c "print('Test output')" >> /root/autodl-tmp/project/output.log 2>&1 &