


# Introduction


# Background


# Problem and solution with features


# Conclusion / Expectation 
 

--------


# How to track bugs
* About issues  
You can use issues to track bugs for work on GitHub. You can collect user feedback, report software bugs, and organize tasks you'd like to accomplish with issues in a repository. Issues can act as more than just a place to report software bugs. To stay updated on the most recent comments in an issue, you can watch an issue to receive notifications about the latest comments. With issues, you can do:  
1. Track and prioritize your work using project boards.  
2. Associate issues with pull requests so that your issue automatically closes when you merge a pull request.  
3. Track duplicate issues using saved replies  
4. Create issue templates to help contributors open meaningful issues. For more information, see the below.  

* Assigning issues  
You can assign up to 10 people to each issue or pull request, including yourself, anyone with write permissions to the repository, and organization members with read permissions to the repository.
1. On GitHub, navigate to the main page of the repository.
2. Under the repository name, click  "Issues".
3. Select the checkbox next to the items you want to assign to someone.
4. In the upper-right corner, click "Assign".
5. To assign the items to a user, start typing their username, then click their name when it appears. You can select and add up to ten assignees to an issue.

# PLEASE FOLLOW THE STEPS BEFORE YOU PUSH/PULL  
Note: All of the git commands are the same across Mac and Windows.

### How to make a new branch
```
git branch //This command lets you check and ensure you are on master (you will be if just cloned the repository)
	
git checkout -b "alex-dev" //This creates and checks out the new branch (include the quotes ""). Change the name alex-dev to whatever you want your branch to be called.

git branch //This confirms that you are now working on your new branch
		
git status //This shows you what has been modified since the last commit. It should say nothing to commit right now.
```
### When you push, 
```
 git branch //This confirms that you are now working on your new branch
 git status
 git add -A
 git commit -m "Your commit message"
 git push origin urName-dev
```
### When you pull,
```
 git branch //it should still be your development branch, not master
 git checkout master
 git pull origin master
 git checkout urName-dev
 git merge master
```
# Git rules of thumb
1. Never Work on Master. All work must be done on your own branch.
2. Keep your changes small.
3. If you see something say something.
4. Git does not store empty folders.

# Contribution Guidelines
Please ensure your pull request adheres to the following guidelines:

- Alphabetize your entry.
- Suggested READMEs should be beautiful or stand out in some way.
- Keep descriptions short and simple, but descriptive.
- Start the description with a capital and end with a full stop/period.
- Check your spelling and grammar.
- Make sure your text editor is set to remove trailing whitespace.

# Authors  
  * **Su In Cho, SBU ID: 111669000,**  *Lead Programmer*
  * **Jinwoo Choi, SBU ID: 110096881,**  *Project Manager*
