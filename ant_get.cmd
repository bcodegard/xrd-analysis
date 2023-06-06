: stops the commands that the script runs from showing
: but does not stop the output of echo from showing
@echo off

: loop with run number from first argument to second argument
: inclusive on both ends
for /l %%r in (%1, 1, %2) do (

	: check if file exists by looking at the header
	curl "http://cms2.physics.ucsb.edu/DRS/Run%%r/Run%%r_pulses.ant" --output nul --head --fail --silent && (
		: download the file if it exists
		curl "http://cms2.physics.ucsb.edu/DRS/Run%%r/Run%%r_pulses.ant" --output ".\data\rpi\ant\Run%%r_pulses.ant"
	) || (echo run %%r curl unsucessful)
	echo.
)
