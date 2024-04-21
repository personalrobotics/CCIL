$(document).ready(function() {
    const syncedGroups = ["synced-1", "synced-2"];

    let vidsReady = {};
    let vidsEnded = {};

    for (const group of syncedGroups) {
        vidsReady[group] = 0;
        vidsEnded[group] = 0;

        const syncedVids = document.querySelectorAll(`video.${group}`);

        for (const vid of syncedVids) {
            vid.oncanplay = () => {
                vid.oncanplay = null;
                vidsReady[group] += 1;
                console.log(`${vidsReady[group]}/${syncedVids.length} videos ready in group ${group}`);
                if (vidsReady[group] === syncedVids.length) {
                    console.log(`Starting all videos in group ${group}`);
                    for (const v of syncedVids) {
                        v.play();
                    }
                }
            }

            vid.onended = () => {
                vidsEnded[group] += 1;
                console.log(`${vidsEnded[group]}/${syncedVids.length} videos finished in group ${group}`);
                if (vidsEnded[group] === syncedVids.length) {
                    console.log(`Restarting all videos in group ${group}`);
                    setTimeout(() => {
                        vidsEnded[group] = 0;
                        for (const v of syncedVids) {
                            v.currentTime = 0;
                            v.play();
                        }
                    }, 1000);
                }
            }
        }
    }

    
});
