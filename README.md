This is the repository for the thermal department of DH10. 

Please commit files with proper filenames, and commit messages. Do not commit changes to the main branch without confirming your changes.

*Tips for GitHub* 
- You can download an extension for VSCode to help you manage file access (but the terminal is always faster/reliable)
- To work locally, you must *Clone* the repository. Any changes then made are not shared until you *Commit* (please add a proper commit message) and then *Push*. This will update the repository for everyone
- In general, work on a branch of the repository, this prevents any unrecoverable changes. This is especially true if you are editing a file rather than making a new one.

*To clone a repository through the terminal*
1. Ensure you have Git installed by running: git --version
2. Make a file directory where you  your cloned repository to sit. Navigate to it through Powershell. (*cd* opens a folder, *ls* lists all of its contents, *cd ..* will exit the current folder)
2. Make a file directory where you want your cloned repository to sit. Navigate to it through Powershell. (*cd* opens a folder, *ls* lists all of its contents, *cd ..* will exit the current folder)
3. Get the url of the repository by going through *Code > HTTPS* and copying the link
4. In your Powershell window, type *git clone \<the url\>*
*To push a repository through the terminal*
5. Save all files and then run *git add .* (the dot after add can be replaced dependent on the situation, check documentation)
6. Run *git commit -m "include a relevant message here"*
7. Run *git push origin \<whatever branch you want\>*
