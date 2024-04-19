$(document).ready(function() {
    const syncedVids = document.querySelectorAll('video.synced');

    let vidsReady = 0;
    let vidsEnded = 0;

    for (const vid of syncedVids) {
        vid.oncanplay = () => {
            vid.oncanplay = null;
            vidsReady += 1;
            console.log(`${vidsReady}/${syncedVids.length} videos ready`);
            if (vidsReady === syncedVids.length) {
                console.log("Starting all videos");
                for (const v of syncedVids) {
                    v.play();
                }
            }
        }

        vid.onended = () => {
            vidsEnded += 1;
            console.log(`${vidsEnded}/${syncedVids.length} videos finished`);
            if (vidsEnded === syncedVids.length) {
                console.log("Restarting all videos");
                setTimeout(() => {
                    vidsEnded = 0;
                    for (const v of syncedVids) {
                        v.currentTime = 0;
                        v.play();
                    }
                }, 1000);
            }
        }
    }
});
